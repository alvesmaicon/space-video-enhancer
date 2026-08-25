"""Per-frame cost of the stabilization building blocks on a 720p synthetic scene."""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators, transforms
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, StabilizationMode

WIDTH, HEIGHT = 1280, 720
FOCUS = NormalizedRect(0.3, 0.3, 0.4, 0.4)


def make_frames(count: int, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    canvas = cv2.resize(
        rng.integers(0, 255, (90, 160, 3), dtype=np.uint8), (WIDTH + 200, HEIGHT + 200), interpolation=cv2.INTER_CUBIC
    )
    for _ in range(80):
        x, y = int(rng.integers(0, WIDTH + 150)), int(rng.integers(0, HEIGHT + 150))
        cv2.rectangle(canvas, (x, y), (x + 40, y + 25), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    frames = []
    for _ in range(count):
        dx, dy = rng.normal(0.0, 6.0, 2)
        matrix = np.array([[1.0, 0.0, -(100 + dx)], [0.0, 1.0, -(100 + dy)]], dtype=np.float32)
        frames.append(cv2.warpAffine(canvas, matrix, (WIDTH, HEIGHT)))
    return frames


def timeit(label: str, fn: Callable[[], object], repetitions: int = 10) -> None:
    fn()
    started = time.perf_counter()
    for _ in range(repetitions):
        fn()
    print(f"  {label:60s} {(time.perf_counter() - started) * 1000 / repetitions:7.1f} ms/frame")


def main() -> None:
    frames = make_frames(12)
    translation_state = StabilizationState()
    similarity_state = StabilizationState()
    counter = {"translation": 0, "similarity": 0}

    def translation() -> None:
        counter["translation"] += 1
        index = counter["translation"]
        estimators.estimate_translation_shift(frames[index % len(frames)], index, translation_state)

    def similarity() -> None:
        counter["similarity"] += 1
        index = counter["similarity"]
        estimators.estimate_inter_frame_transform(
            frames[index % len(frames)], index, similarity_state, StabilizationMode.SIMILARITY
        )

    timeit("estimate_translation_shift (GFTT + LK @480 px)", translation)
    timeit("estimate_inter_frame_transform SIMILARITY", similarity)
    timeit(
        "estimate_camera_lock_transform (ORB referencia + atual)",
        lambda: estimators.estimate_camera_lock_transform(frames[1], frames[0], FOCUS),
    )
    reference_gray = estimators.scaled_gray(frames[0], estimators.CAMERA_LOCK_MAX_DIMENSION / WIDTH)
    orb = cv2.ORB_create(**estimators.ORB_PARAMS)
    timeit("  so ORB detectAndCompute do frame de referencia", lambda: orb.detectAndCompute(reference_gray, None))
    transform = transforms.translation_to_affine(5.3, -3.1)
    timeit("warp affine 720p", lambda: transforms.warp(frames[0], transform))
    timeit(
        "auto_crop 720p (mascara + erode + boundingRect + resize)", lambda: transforms.auto_crop(frames[0], transform)
    )

    rng = np.random.default_rng(1)
    zooms = []
    for _ in range(40):
        dx, dy = rng.normal(0.0, 6.0, 2)
        mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        valid = cv2.warpAffine(
            mask,
            transforms.translation_to_affine(dx, dy),
            (WIDTH, HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        _x, _y, cropped_width, _height = cv2.boundingRect(cv2.findNonZero(valid))
        zooms.append(WIDTH / cropped_width)
    print(
        f"  auto_crop por frame com tremor std 6 px: zoom min {min(zooms):.4f} max {max(zooms):.4f} "
        f"(variacao {100 * (max(zooms) - min(zooms)):.2f}% frame a frame)"
    )


if __name__ == "__main__":
    main()
