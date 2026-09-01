"""Diagnostico do padrao de tremor de um video real: mede e recomenda a combinacao de correcoes.

Uma passada de rastreamento (mesmo estimador robusto dos modos automaticos, no tamanho de analise)
coleta, por par de frames: deslocamento do centro, rotacao, zoom, gradiente do deslocamento ao longo
de cada eixo de leitura (assinatura de rolling shutter) e, em pares amostrados, quanto uma homografia
explica alem da similaridade (conteudo de perspectiva). Do caminho acumulado sai o espectro do tremor
(banda lenta = caminhada/deriva, media = balanco da mao, alta = vibracao); dos gradientes sai a fracao
de leitura do sensor (o valor do slider de rolling shutter, estimado por regressao contra a aceleracao,
nao chutado) e a energia de jello restante (o que a correcao automatica de oscilacao removeria).

O relatorio final mapeia cada medida para os controles reais da aba Estabilizar: modo, suavizacao,
slider de rolling shutter (%), oscilacao automatica e eixo de leitura — mais um aviso quando o
video parece entrelacado (o app nao desentrelaca; use ffmpeg antes).

Exemplos:

    python benchmarks/diagnose_shake.py video.mp4
    python benchmarks/diagnose_shake.py video.mp4 --first 120 --last 600 --json relatorio.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from space_video_enhancer.pipeline.stabilization.estimators import analysis_gray
from space_video_enhancer.pipeline.stabilization.global_motion import (
    estimate_global_motion,
    new_cell_memory,
    upgrade_to_homography,
)
from space_video_enhancer.pipeline.stabilization.rolling_shutter import center_displacement, motion_gradient
from space_video_enhancer.pipeline.stabilization.stabilize import (
    SMOOTHING_MAX_FRAMES,
    SMOOTHING_MIN_FRAMES,
    smoothing_time_constant,
)
from space_video_enhancer.pipeline.stabilization.wobble import (
    AXIS_MIN_CORRELATION,
    detect_readout_axis,
    filter_wobble,
    readout_correlation,
    wobble_window_frames,
)
from space_video_enhancer.settings import ReadoutAxis, StabilizationMode
from space_video_enhancer.video import VideoSource

DEFAULT_MAX_FRAMES = 900
"""Sem --last, analisa ate isso (~30 s a 30 fps): o padrao de tremor nao muda depois disso."""
HOMOGRAPHY_SAMPLE_STRIDE = 4
"""A perspectiva e medida a cada tantos pares (o ajuste de homografia e o passo caro)."""
COMB_SAMPLES = 8

ROTATION_SIGNIFICANT_DEG = 0.10
"""RMS de rotacao residual acima disso: o modo so-translacao deixa o quadro 'girando'."""
ZOOM_SIGNIFICANT = 0.002
"""RMS de log-zoom residual acima disso (~0,2%): a variacao de tamanho entre frames aparece."""
PERSPECTIVE_SIGNIFICANT_PX = 0.6
"""Mediana do residuo de perspectiva (px de fonte por par) acima disso: vale homografia."""
PERSPECTIVE_MIN_UPGRADES = 0.5
"""...e a homografia precisa ter sido suportada em pelo menos metade dos pares amostrados."""
JELLO_SIGNIFICANT_PX = 0.8
"""RMS de jello (px de fonte) acima disso: ligar a compensacao automatica de oscilacao."""
TRANSLATION_CAP_FRACTION = 0.05
"""Teto de translacao dos modos automaticos (stabilize.py): tremor alem disso satura a correcao."""
SMOOTHING_PERIODS = 1.5
"""A constante de tempo da camera virtual cobre tantas vezes o periodo dominante do tremor."""
MIN_MEASURED_FRACTION = 0.7

BANDS = (
    ("lenta (<1,5 Hz — caminhada/deriva)", 0.0, 1.5),
    ("media (1,5-5 Hz — balanco da mao)", 1.5, 5.0),
    ("alta (>5 Hz — tremor/vibracao)", 5.0, 1e9),
)


COMB_THRESHOLD = 24
LIKELY_INTERLACED_SCORE = 0.01
"""Deteccao de entrelacamento (familia decomb/yadif): uma linha e 'pente' quando difere das duas
vizinhas na mesma direcao, (a-b)*(c-b) > T^2. So diagnostico — o app nao desentrelaca."""


def comb_score(frame: np.ndarray) -> float:
    """Fracao amostrada de pixels com pente: ~0 em video progressivo, porcentos em entrelacado com movimento."""
    height = frame.shape[0]
    if height < 4:
        return 0.0
    luma = (frame[..., 1] if frame.ndim == 3 else frame).astype(np.int32)
    rows = np.arange(1, height - 1, 8)
    columns = slice(None, None, 4)
    above, row, below = luma[rows - 1, columns], luma[rows, columns], luma[rows + 1, columns]
    mask = (above - row) * (below - row) > COMB_THRESHOLD * COMB_THRESHOLD
    return float(mask.mean()) if mask.size else 0.0


def likely_interlaced(score: float) -> bool:
    return score >= LIKELY_INTERLACED_SCORE


def hann_trend(values: np.ndarray, half: int) -> np.ndarray:
    """Media movel centrada com janela de Hann (mesmo truque de filter_wobble), unilateral nas pontas."""
    count = len(values)
    kernel = np.hanning(2 * half + 3)[1:-1]
    kernel /= kernel.sum()
    weight = np.convolve(np.pad(np.ones(count), half), kernel, mode="valid")
    return np.convolve(np.pad(values, half), kernel, mode="valid") / weight


def band_rms(residual: np.ndarray, fps: float) -> tuple[list[tuple[str, float]], float]:
    """RMS do residuo por banda de frequencia e a frequencia dominante (Hz), via rFFT com janela de Hann."""
    count = len(residual)
    if count < 8:
        return [(label, 0.0) for label, _low, _high in BANDS], 0.0
    window = np.hanning(count)
    spectrum = np.fft.rfft(residual * window)
    frequencies = np.fft.rfftfreq(count, d=1.0 / fps)
    # Energia por raia normalizada pela janela: somada por banda e devolvida como RMS equivalente.
    energy = (np.abs(spectrum) ** 2) * 2.0 / (np.sum(window**2) * count)
    energy[0] /= 2.0
    per_band = []
    for label, low, high in BANDS:
        selected = (frequencies >= low) & (frequencies < high)
        per_band.append((label, float(math.sqrt(max(0.0, energy[selected].sum())))))
    meaningful = frequencies >= 0.5
    dominant = float(frequencies[meaningful][np.argmax(np.abs(spectrum[meaningful]))]) if meaningful.any() else 0.0
    return per_band, dominant


def readout_fraction(changes: np.ndarray, displacements: np.ndarray, measured: np.ndarray) -> float:
    """Fracao de leitura do sensor por regressao (pela origem) dos gradientes contra a aceleracao.

    A distorcao de um frame e r vezes a velocidade durante ele, entao a mudanca entre dois frames
    e ``r * (v_k - v_{k-1})``: o coeficiente da reta gradiente x aceleracao E o valor do slider.
    """
    both = measured[1:] & measured[:-1]
    if int(both.sum()) < 8:
        return 0.0
    acceleration = (displacements[1:] - displacements[:-1])[both].ravel()
    gradient = changes[1:][both].ravel()
    denominator = float(np.dot(acceleration, acceleration))
    if denominator < 1e-9:
        return 0.0
    return float(np.clip(np.dot(gradient, acceleration) / denominator, 0.0, 1.0))


def smoothing_slider_for(time_constant: float) -> float:
    """Inverso de smoothing_time_constant: o valor do slider (0..1) que da essa constante de tempo."""
    ratio = SMOOTHING_MAX_FRAMES / SMOOTHING_MIN_FRAMES
    clamped = min(max(time_constant, SMOOTHING_MIN_FRAMES), SMOOTHING_MAX_FRAMES)
    return math.log(clamped / SMOOTHING_MIN_FRAMES) / math.log(ratio)


def analyze(path: Path, first: int, last: int) -> dict:
    source = VideoSource(path)
    last = min(last, source.frame_count - 1)
    count = last - first + 1
    if count < 10:
        raise SystemExit(f"Trecho curto demais para diagnostico: {count} frames")

    shifts = np.zeros((count, 2), dtype=np.float64)  # deslocamento da cena no centro, px de analise
    rotations = np.zeros(count, dtype=np.float64)  # rotacao por par, graus
    log_scales = np.zeros(count, dtype=np.float64)
    row_changes = np.zeros((count, 2), dtype=np.float64)
    column_changes = np.zeros((count, 2), dtype=np.float64)
    measured = np.zeros(count, dtype=bool)
    perspective_residuals: list[float] = []
    homography_attempts = 0
    homography_upgrades = 0
    comb_scores: list[float] = []
    comb_at = {first + index * max(1, (count - 1) // max(1, COMB_SAMPLES - 1)) for index in range(COMB_SAMPLES)}

    memory = new_cell_memory()
    previous_gray: np.ndarray | None = None
    analysis_scale = 1.0
    analysis_size = (0, 0)
    frames_read = 0

    for offset in range(count):
        frame = source.read_frame(first + offset)
        if frame is None:
            break
        frames_read += 1
        if first + offset in comb_at:
            comb_scores.append(comb_score(frame))
        gray, scale = analysis_gray(frame)
        analysis_scale = scale
        if gray.size:
            analysis_size = (int(gray.shape[1]), int(gray.shape[0]))
        if previous_gray is not None and gray.shape == previous_gray.shape and gray.size:
            motion = estimate_global_motion(previous_gray, gray, StabilizationMode.SIMILARITY, memory)
            if motion is not None:
                measured[offset] = True
                transform = motion.transform
                shifts[offset] = center_displacement(transform, gray.shape)
                rotations[offset] = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
                log_scales[offset] = math.log(max(1e-6, math.hypot(transform[0, 0], transform[1, 0])))
                width, height = analysis_size
                along_rows = motion_gradient(motion.vectors, height, ReadoutAxis.ROWS)
                if along_rows is not None:
                    row_changes[offset] = along_rows
                along_columns = motion_gradient(motion.vectors, width, ReadoutAxis.COLUMNS)
                if along_columns is not None:
                    column_changes[offset] = along_columns
                if offset % HOMOGRAPHY_SAMPLE_STRIDE == 0:
                    homography_attempts += 1
                    vectors = motion.vectors
                    homography = upgrade_to_homography(vectors.previous, vectors.current, vectors.inliers, vectors.size)
                    if homography is not None:
                        homography_upgrades += 1
                        perspective_residuals.append(_corner_divergence(transform, homography, analysis_size))
        previous_gray = gray
        done = offset + 1
        if done % 120 == 0 or done == count:
            sys.stderr.write(f"\rrastreando: {done}/{count} frames")
            sys.stderr.flush()
    sys.stderr.write("\n")

    if frames_read < 10:
        raise SystemExit(f"So {frames_read} frames decodificados; diagnostico impossivel")
    for series in (shifts, rotations, log_scales, row_changes, column_changes, measured):
        series.resize((frames_read, *series.shape[1:]), refcheck=False)

    to_source = 1.0 / max(1e-6, analysis_scale)
    half = wobble_window_frames(source.fps)

    # Caminho acumulado e residuo rapido (o tremor) apos remover a tendencia de ~1 s (o movimento intencional).
    path_x = np.cumsum(shifts[:, 0]) * to_source
    path_y = np.cumsum(shifts[:, 1]) * to_source
    residual_x = path_x - hann_trend(path_x, half)
    residual_y = path_y - hann_trend(path_y, half)
    residual = np.hypot(residual_x, residual_y)
    bands_x, dominant_x = band_rms(residual_x, source.fps)
    bands_y, dominant_y = band_rms(residual_y, source.fps)
    shake_rms_px = float(np.sqrt(np.mean(residual**2)))
    shake_peak_px = float(np.max(residual)) if len(residual) else 0.0
    dominant_hz = dominant_x if np.sqrt(np.mean(residual_x**2)) >= np.sqrt(np.mean(residual_y**2)) else dominant_y

    rotation_path = np.cumsum(rotations)
    rotation_rms = float(np.sqrt(np.mean((rotation_path - hann_trend(rotation_path, half)) ** 2)))
    zoom_path = np.cumsum(log_scales)
    zoom_rms = float(np.sqrt(np.mean((zoom_path - hann_trend(zoom_path, half)) ** 2)))

    correlation_rows = readout_correlation(row_changes, shifts, measured)
    correlation_columns = readout_correlation(column_changes, shifts, measured)
    axis = detect_readout_axis(correlation_rows, correlation_columns, source.rotation)
    axis_changes = column_changes if axis is ReadoutAxis.COLUMNS else row_changes
    axis_length = analysis_size[0] if axis is ReadoutAxis.COLUMNS else analysis_size[1]
    readout = readout_fraction(axis_changes, shifts, measured)
    axis_correlation = correlation_columns if axis is ReadoutAxis.COLUMNS else correlation_rows
    wobble = filter_wobble(axis_changes / axis_length, shifts / axis_length, readout, half)
    jello_rms_px = float(np.sqrt(np.mean(np.sum(wobble**2, axis=1)))) * axis_length * to_source

    return {
        "video": str(path),
        "fps": source.fps,
        "size": [source.width, source.height],
        "rotation_metadata": source.rotation,
        "frames": [first, first + frames_read - 1],
        "measured_fraction": float(measured.mean()),
        "shake_rms_px": shake_rms_px,
        "shake_peak_px": shake_peak_px,
        "shake_rms_fraction": shake_rms_px / max(1, source.height),
        "shake_peak_fraction": shake_peak_px / max(1, source.height),
        "dominant_hz": dominant_hz,
        "bands_x_px": bands_x,
        "bands_y_px": bands_y,
        "rotation_rms_deg": rotation_rms,
        "zoom_rms": zoom_rms,
        "perspective_residual_px": (
            float(np.median(perspective_residuals)) * to_source if perspective_residuals else 0.0
        ),
        "homography_support": (homography_upgrades / homography_attempts if homography_attempts else 0.0),
        "readout_axis": axis.value,
        "readout_axis_correlation": axis_correlation,
        "readout_fraction": readout,
        "jello_rms_px": jello_rms_px,
        "comb_score_max": max(comb_scores) if comb_scores else 0.0,
    }


def _corner_divergence(similarity: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> float:
    """Mediana (px de analise) do quanto a homografia move os cantos alem da similaridade: so perspectiva."""
    width, height = size
    corners = np.array(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [0.0, height, 1.0], [width, height, 1.0]], dtype=np.float64
    ).T
    affine = np.vstack([similarity.astype(np.float64), (0.0, 0.0, 1.0)])
    by_similarity = affine @ corners
    by_homography = homography.astype(np.float64) @ corners
    by_homography = by_homography / np.maximum(1e-9, np.abs(by_homography[2])) * np.sign(by_homography[2])
    return float(np.median(np.hypot(*(by_homography[:2] - by_similarity[:2]))))


def recommend(report: dict) -> list[str]:
    """Mapa medida -> controle da aba Estabilizar, com o porque de cada escolha."""
    lines = []
    fps = report["fps"]

    if report["measured_fraction"] < MIN_MEASURED_FRACTION:
        lines.append(
            f"ATENCAO: o rastreador so mediu {report['measured_fraction']:.0%} dos pares — cena dificil "
            "(desfoque/pouca textura). Os automaticos vao 'segurar' nesses trechos; se o resultado deformar, "
            "o caminho e o modo Manual + 'Preencher com o movimento'."
        )

    perspective = (
        report["perspective_residual_px"] >= PERSPECTIVE_SIGNIFICANT_PX
        and report["homography_support"] >= PERSPECTIVE_MIN_UPGRADES
    )
    rotation = report["rotation_rms_deg"] >= ROTATION_SIGNIFICANT_DEG
    zoom = report["zoom_rms"] >= ZOOM_SIGNIFICANT
    if perspective:
        mode = "Perspectiva"
        why = (
            f"residuo de perspectiva {report['perspective_residual_px']:.2f} px/par com homografia "
            f"suportada em {report['homography_support']:.0%} dos pares"
        )
    elif rotation or zoom:
        mode = "Suavizar tremor + rotacao e zoom"
        parts = []
        if rotation:
            parts.append(f"rotacao residual {report['rotation_rms_deg']:.2f}° RMS")
        if zoom:
            parts.append(f"zoom residual {report['zoom_rms'] * 100:.2f}% RMS")
        why = " e ".join(parts)
    else:
        mode = "Suavizar tremor (translacao)"
        why = (
            f"rotacao ({report['rotation_rms_deg']:.2f}°) e zoom ({report['zoom_rms'] * 100:.2f}%) abaixo do que "
            "aparece na imagem; o modo mais simples deforma menos"
        )
    lines.append(f"Modo: {mode} — {why}.")

    dominant = max(0.5, report["dominant_hz"])
    time_constant = SMOOTHING_PERIODS * fps / dominant
    slider = smoothing_slider_for(time_constant)
    lines.append(
        f"Suavizacao: ~{slider:.0%} (constante de tempo ≈ {smoothing_time_constant(slider):.0f} frames) — "
        f"tremor dominante em {report['dominant_hz']:.1f} Hz; a camera virtual precisa ser mais lenta que ele."
    )

    if report["readout_axis_correlation"] >= AXIS_MIN_CORRELATION and report["readout_fraction"] > 0.02:
        lines.append(
            f"Rolling shutter: slider em {report['readout_fraction'] * 100:.0f}% (fracao de leitura MEDIDA por "
            f"regressao, correlacao {report['readout_axis_correlation']:.2f}) — eixo de leitura: "
            f"{report['readout_axis']} (deixe em Auto, a deteccao concorda)."
        )
    else:
        lines.append(
            "Rolling shutter: slider em 0% — a assinatura (gradiente seguindo a aceleracao) nao apareceu "
            f"(correlacao {report['readout_axis_correlation']:.2f}); subir o slider aqui so distorceria."
        )

    if report["jello_rms_px"] >= JELLO_SIGNIFICANT_PX:
        lines.append(
            f"Compensar a oscilacao automaticamente: LIGAR — jello de {report['jello_rms_px']:.1f} px RMS "
            "alem do que o slider explica."
        )
    else:
        lines.append(
            f"Compensar a oscilacao automaticamente: desnecessario "
            f"(jello residual {report['jello_rms_px']:.1f} px RMS)."
        )

    if likely_interlaced(report["comb_score_max"]):
        lines.append(
            f"ATENCAO: pente de entrelacamento detectado (comb score {report['comb_score_max']:.3f}) — "
            "desentrelace fora do app (ex.: ffmpeg -vf bwdif) antes de estabilizar."
        )

    if report["shake_peak_fraction"] > TRANSLATION_CAP_FRACTION:
        lines.append(
            f"ATENCAO: picos de tremor de {report['shake_peak_fraction']:.1%} do quadro passam do teto de 5% dos "
            "modos automaticos — vai sobrar movimento nos piores frames. Opcoes: aceitar o residuo, ou modo "
            "Manual + 'Preencher com o movimento' (sem teto) com keyframes a cada ~10 frames."
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--first", type=int, default=0, help="primeiro frame do trecho analisado")
    parser.add_argument(
        "--last", type=int, default=None, help=f"ultimo frame (padrao: first + {DEFAULT_MAX_FRAMES - 1})"
    )
    parser.add_argument("--json", type=Path, default=None, help="grava o relatorio bruto em JSON")
    arguments = parser.parse_args()

    last = arguments.last if arguments.last is not None else arguments.first + DEFAULT_MAX_FRAMES - 1
    report = analyze(arguments.video, arguments.first, last)

    print(f"\n=== Diagnostico de tremor: {arguments.video.name} ===")
    print(
        f"{report['size'][0]}x{report['size'][1]} @ {report['fps']:.2f} fps, "
        f"frames {report['frames'][0]}-{report['frames'][1]}, "
        f"rastreado em {report['measured_fraction']:.0%} dos pares"
    )
    print("\nTremor (residuo apos remover o movimento intencional de ~1 s):")
    print(
        f"  RMS {report['shake_rms_px']:.1f} px ({report['shake_rms_fraction']:.2%} do quadro), "
        f"pico {report['shake_peak_px']:.1f} px"
    )
    print(f"  frequencia dominante: {report['dominant_hz']:.1f} Hz")
    for axis_label, bands in (("X", report["bands_x_px"]), ("Y", report["bands_y_px"])):
        for label, value in bands:
            print(f"  {axis_label} banda {label}: {value:.2f} px RMS")
    print(
        f"  rotacao residual: {report['rotation_rms_deg']:.3f}° RMS | "
        f"zoom residual: {report['zoom_rms'] * 100:.2f}% RMS"
    )
    print(
        f"  perspectiva: {report['perspective_residual_px']:.2f} px/par (homografia suportada em "
        f"{report['homography_support']:.0%} dos pares amostrados)"
    )
    print(
        f"  rolling shutter: eixo {report['readout_axis']} (correlacao {report['readout_axis_correlation']:.2f}), "
        f"fracao de leitura estimada {report['readout_fraction'] * 100:.0f}%, jello alem do slider "
        f"{report['jello_rms_px']:.1f} px RMS"
    )
    print(f"  entrelacamento: comb score maximo {report['comb_score_max']:.3f}")

    print("\n=== Receita recomendada (aba Estabilizar) ===")
    for line in recommend(report):
        print(f"  • {line}")

    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRelatorio bruto: {arguments.json}")


if __name__ == "__main__":
    main()
