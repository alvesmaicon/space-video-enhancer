#!/usr/bin/env python3
"""Convert security-camera ``.dav`` recordings (Dahua DVR/NVR) to ``.mp4``.

A ``.dav`` is usually H.264/H.265 video (and AAC/G.711 audio) in Dahua's own container, which
ffmpeg reads with its ``dhav`` demuxer. The fast path just remuxes the streams into MP4 (no
quality loss, seconds per file); when that is not possible — a codec MP4 does not carry, broken
timestamps, an output shorter than the recording — the file is re-encoded (H.264 + AAC).

    python3 scripts/dav_to_mp4.py ~/Downloads/my/videos/            # every .dav in the folder
    python3 scripts/dav_to_mp4.py camera.dav -o ~/Movies/             # one file, another folder
    python3 scripts/dav_to_mp4.py *.dav --reencode                    # always re-encode

Outputs go next to each input (or to ``--output-dir``) with the same name and ``.mp4``; an
existing output is skipped unless ``--overwrite``. Needs ``ffmpeg`` and ``ffprobe`` on the PATH
(or ``--ffmpeg``); Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

COPYABLE_VIDEO = {"h264", "hevc"}
COPYABLE_AUDIO = {"aac", "mp3"}
"""Codecs MP4 carries as they are; anything else (G.711 ``pcm_alaw``/``pcm_mulaw``, ...) is re-encoded."""
DURATION_TOLERANCE = 0.02
"""A remux shorter than the recording by more than this fraction (or 1 s) is treated as failed."""


def probe(ffprobe: str, path: Path) -> dict:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe falhou em {path.name}")
    return json.loads(result.stdout)


def duration_of(info: dict) -> float:
    try:
        return float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


def codecs(info: dict, kind: str) -> list[str]:
    return [stream.get("codec_name", "") for stream in info.get("streams", []) if stream.get("codec_type") == kind]


def run_ffmpeg(ffmpeg: str, arguments: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-y", *arguments], capture_output=True, text=True, check=False
    )
    return result.returncode == 0, result.stderr.strip()


def remux(ffmpeg: str, source: Path, target: Path, has_audio: bool) -> tuple[bool, str]:
    arguments = ["-fflags", "+genpts", "-i", str(source), "-map", "0:v:0"]
    if has_audio:
        arguments += ["-map", "0:a:0"]
    arguments += ["-c", "copy", "-movflags", "+faststart", str(target)]
    return run_ffmpeg(ffmpeg, arguments)


def reencode(ffmpeg: str, source: Path, target: Path, has_audio: bool, crf: int) -> tuple[bool, str]:
    arguments = ["-fflags", "+genpts", "-i", str(source), "-map", "0:v:0"]
    if has_audio:
        arguments += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "96k"]
    arguments += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    return run_ffmpeg(ffmpeg, arguments)


def output_is_complete(ffprobe: str, target: Path, expected_duration: float) -> tuple[bool, str]:
    """The MP4 opens, has video, and is about as long as the recording."""
    if not target.exists() or target.stat().st_size == 0:
        return False, "arquivo de saída vazio"
    try:
        info = probe(ffprobe, target)
    except RuntimeError as error:
        return False, str(error)
    if not codecs(info, "video"):
        return False, "saída sem vídeo"
    duration = duration_of(info)
    if expected_duration > 0 and duration < expected_duration - max(1.0, expected_duration * DURATION_TOLERANCE):
        return False, f"saída com {duration:.1f}s de {expected_duration:.1f}s"
    return True, f"{duration:.1f}s"


def convert(
    source: Path,
    output_dir: Path | None,
    *,
    ffmpeg: str,
    ffprobe: str,
    force_reencode: bool,
    crf: int,
    overwrite: bool,
) -> bool:
    target = (output_dir or source.parent) / f"{source.stem}.mp4"
    if target.exists() and not overwrite:
        print(f"  pulado: {target.name} já existe (use --overwrite)")
        return True
    started = time.perf_counter()
    try:
        info = probe(ffprobe, source)
    except RuntimeError as error:
        print(f"  ERRO: não deu para ler {source.name}: {error}")
        return False
    video, audio = codecs(info, "video"), codecs(info, "audio")
    if not video:
        print(f"  ERRO: {source.name} não tem trilha de vídeo")
        return False
    expected = duration_of(info)
    has_audio = bool(audio)
    summary = f"{video[0]}" + (f" + {audio[0]}" if has_audio else ", sem áudio") + f", {expected:.0f}s"
    print(f"  {summary}")

    partial = target.with_name(f".{target.name}.part.mp4")  # never leave a half-written file under the real name
    copyable = video[0] in COPYABLE_VIDEO and (not has_audio or audio[0] in COPYABLE_AUDIO)
    method = None
    if copyable and not force_reencode:
        ok, error = remux(ffmpeg, source, partial, has_audio)
        complete, detail = output_is_complete(ffprobe, partial, expected) if ok else (False, error)
        if complete:
            method = "remux (sem recodificar)"
        else:
            print(f"  remux não serviu ({detail.splitlines()[-1] if detail else 'erro'}); recodificando...")
    if method is None:
        ok, error = reencode(ffmpeg, source, partial, has_audio, crf)
        complete, detail = output_is_complete(ffprobe, partial, expected) if ok else (False, error)
        if not complete:
            partial.unlink(missing_ok=True)
            print(f"  ERRO: falhou ao recodificar: {detail}")
            return False
        method = f"recodificado (H.264 crf {crf})"
    partial.replace(target)
    size_mb = target.stat().st_size / 1_000_000
    print(f"  ok: {target.name} — {method}, {detail}, {size_mb:.1f} MB em {time.perf_counter() - started:.1f}s")
    return True


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(item for item in path.iterdir() if item.suffix.lower() == ".dav" and item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            print(f"aviso: {path} não existe", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Converte gravações .dav (câmera de segurança) para .mp4.")
    parser.add_argument("paths", nargs="+", type=Path, help="arquivos .dav ou pastas com eles")
    parser.add_argument("-o", "--output-dir", type=Path, help="pasta de saída (padrão: a mesma de cada arquivo)")
    parser.add_argument("--reencode", action="store_true", help="sempre recodificar (padrão: só copia as trilhas)")
    parser.add_argument("--crf", type=int, default=20, help="qualidade da recodificação, menor = melhor (padrão 20)")
    parser.add_argument("--overwrite", action="store_true", help="sobrescrever .mp4 que já existam")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="caminho do ffmpeg (o ffprobe é procurado ao lado)")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        print("ffmpeg não encontrado (brew install ffmpeg, ou --ffmpeg /caminho/ffmpeg)", file=sys.stderr)
        return 2
    sibling = Path(ffmpeg).with_name("ffprobe")
    ffprobe = str(sibling) if sibling.exists() else shutil.which("ffprobe")
    if ffprobe is None:
        print("ffprobe não encontrado (vem junto com o ffmpeg)", file=sys.stderr)
        return 2

    files = collect([path.expanduser() for path in args.paths])
    if not files:
        print("nenhum arquivo .dav encontrado", file=sys.stderr)
        return 1
    output_dir = args.output_dir.expanduser() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, source in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {source.name}")
        ok = convert(
            source,
            output_dir,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            force_reencode=args.reencode,
            crf=args.crf,
            overwrite=args.overwrite,
        )
        failures += 0 if ok else 1
    print(f"\n{len(files) - failures} de {len(files)} convertido(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
