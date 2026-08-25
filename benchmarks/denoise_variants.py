"""Speed and PSNR of denoise variants on a real frame.

The clean reference is, by default, the temporal median of ``--median-window`` frames around
``--frame`` (valid for a static camera: the median removes the noise, the frame itself keeps
it). With ``--synthetic-noise SIGMA`` the frame is taken as clean and gaussian noise is added
instead. PSNR is measured against that reference; higher is better, but on smooth content plain
blur scores well too, so always look at speed and PSNR together and check the images by eye.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import cv2
import numpy as np

from space_video_enhancer.pipeline.denoise import estimate_denoise_strengths
from space_video_enhancer.settings import DEFAULT_DENOISE_SAMPLE_RECT, DenoiseMode


def read_frames(path: str, start: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"nao foi possivel abrir {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit("nenhum frame lido")
    return frames


def bench(label: str, fn: Callable[[], object], clean: np.ndarray, repetitions: int = 3) -> None:
    fn()
    timings = []
    output: object = None
    for _ in range(repetitions):
        started = time.perf_counter()
        output = fn()
        timings.append((time.perf_counter() - started) * 1000)
    image = output.get() if isinstance(output, cv2.UMat) else output
    assert isinstance(image, np.ndarray)
    print(f"  {label:52s} {min(timings):8.1f} ms   PSNR vs limpo {cv2.PSNR(clean, image):5.2f} dB")


def run_variants(noisy: np.ndarray, clean: np.ndarray, neighbours: list[np.ndarray], h: float, hc: float) -> None:
    height, width = noisy.shape[:2]
    print(f"  h={h:.2f} hc={hc:.2f}   PSNR do frame ruidoso: {cv2.PSNR(clean, noisy):5.2f} dB")
    cv2.ocl.setUseOpenCL(True)
    umat = cv2.UMat(noisy)
    bench(
        "NLMeansColored OpenCL tw=7 sw=21 (atual)",
        lambda: cv2.fastNlMeansDenoisingColored(umat, None, h, hc, 7, 21),
        clean,
    )
    bench("NLMeansColored OpenCL tw=7 sw=15", lambda: cv2.fastNlMeansDenoisingColored(umat, None, h, hc, 7, 15), clean)
    bench("NLMeansColored OpenCL tw=5 sw=11", lambda: cv2.fastNlMeansDenoisingColored(umat, None, h, hc, 5, 11), clean)

    def luma_only() -> cv2.UMat:
        luma, a, b = cv2.split(cv2.cvtColor(umat, cv2.COLOR_BGR2Lab))
        luma = cv2.fastNlMeansDenoising(luma, None, h, 7, 21)
        return cv2.cvtColor(cv2.merge([luma, a, b]), cv2.COLOR_Lab2BGR)

    bench("NLMeans so luma OpenCL (Lab L) sw=21", luma_only, clean)

    def half_resolution() -> cv2.UMat:
        small = cv2.resize(umat, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        denoised = cv2.fastNlMeansDenoisingColored(small, None, h, hc, 7, 21)
        return cv2.resize(denoised, (width, height), interpolation=cv2.INTER_LINEAR)

    bench("NLMeansColored OpenCL a 1/2 res + upscale", half_resolution, clean)
    cv2.ocl.setUseOpenCL(False)
    if len(neighbours) >= 3:
        middle = len(neighbours) // 2
        bench(
            "NLMeansColoredMulti CPU temporal=3",
            lambda: cv2.fastNlMeansDenoisingColoredMulti(neighbours, middle, 3, None, h, hc, 7, 21),
            clean,
            repetitions=1,
        )
    bench("bilateralFilter d=7 sigmaColor=3h sigmaSpace=5", lambda: cv2.bilateralFilter(noisy, 7, 3 * h, 5), clean)
    bench("bilateralFilter d=9 sigmaColor=5h sigmaSpace=7", lambda: cv2.bilateralFilter(noisy, 9, 5 * h, 7), clean)
    bench("GaussianBlur sigma=1.0", lambda: cv2.GaussianBlur(noisy, (0, 0), 1.0), clean)
    bench("sem denoise", lambda: noisy, clean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0, help="indice do frame analisado")
    parser.add_argument("--median-window", type=int, default=15, help="frames na mediana temporal (camera parada)")
    parser.add_argument("--synthetic-noise", type=float, default=None, help="sigma do ruido gaussiano a adicionar")
    parser.add_argument("--h", type=float, default=None, help="forca de luma; padrao: estimativa automatica do app")
    parser.add_argument("--hc", type=float, default=None, help="forca de cor; padrao: estimativa automatica do app")
    args = parser.parse_args()

    window = max(3, args.median_window | 1)
    frames = read_frames(args.video, args.frame - window // 2, window)
    frame = frames[min(len(frames) - 1, window // 2)]
    if args.synthetic_noise is not None:
        rng = np.random.default_rng(1)
        clean = frame
        noise = rng.normal(0.0, args.synthetic_noise, frame.shape[:2])[..., None] + rng.normal(
            0.0, args.synthetic_noise * 0.5, frame.shape
        )
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        neighbours = [np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8) for f in frames]
        print(f"frame {frame.shape[1]}x{frame.shape[0]}, ruido sintetico sigma={args.synthetic_noise:g}")
    else:
        clean = np.median(np.stack(frames), axis=0).astype(np.uint8)
        noisy = frame
        neighbours = frames
        print(f"frame {frame.shape[1]}x{frame.shape[0]}, referencia = mediana de {len(frames)} frames")
    estimated_h, estimated_hc = estimate_denoise_strengths(noisy, DenoiseMode.AUTO, DEFAULT_DENOISE_SAMPLE_RECT)
    run_variants(noisy, clean, neighbours, args.h or estimated_h, args.hc or estimated_hc)


if __name__ == "__main__":
    main()
