"""Residual jitter of each stabilization mode on a synthetic hand-held sequence.

A textured canvas is seen through a window that jitters (gaussian, ``--jitter`` px) and pans
slowly to the right (``--pan`` px over the clip). The position of every output frame is measured
by phase correlation against the first input frame; "jitter" is the spread around the straight
line joining the first and last positions, so a preserved pan does not count as shake.

``--object FRACTION`` adds a large textured block (that fraction of the frame) walking across a
plain background: the estimators must follow the camera, not the block. The report then also
counts how often each auto mode's inter-frame estimate landed on the camera motion.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators
from space_video_enhancer.pipeline.stabilization.stabilize import apply_stabilization
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings

FOCUS = NormalizedRect(0.3, 0.3, 0.4, 0.4)


def make_sequence(count: int, jitter: float, pan: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    width, height = 640, 360
    canvas = cv2.resize(
        rng.integers(0, 255, (60, 100, 3), dtype=np.uint8), (width + 240, height + 240), interpolation=cv2.INTER_CUBIC
    )
    for _ in range(40):
        x, y = int(rng.integers(0, width + 200)), int(rng.integers(0, height + 200))
        cv2.rectangle(canvas, (x, y), (x + 30, y + 20), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    drift = np.stack([np.linspace(0.0, pan, count), np.zeros(count)], axis=1)
    shake = rng.normal(0.0, jitter, (count, 2))
    shake[0] = 0.0  # the first frame sits on the intended path: the virtual camera starts there
    positions = 120.0 + shake + drift
    frames = []
    for x, y in positions:
        matrix = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]], dtype=np.float32)
        frames.append(cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR))
    return frames


OBJECT_SPEED_PX = 6


def make_object_sequence(
    count: int, jitter: float, pan: float, seed: int, object_fraction: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Plain gradient with a few dots (the background) and a big textured block walking right; returns true deltas."""
    rng = np.random.default_rng(seed)
    width, height = 640, 360
    canvas = np.zeros((height + 200, width + 200, 3), np.uint8)
    canvas[..., 0] = np.linspace(60, 140, width + 200, dtype=np.uint8)[None, :]
    canvas[..., 1] = np.linspace(80, 160, height + 200, dtype=np.uint8)[:, None]
    canvas[..., 2] = 120
    for _ in range(40):
        x, y = int(rng.integers(0, width + 190)), int(rng.integers(0, height + 190))
        cv2.circle(canvas, (x, y), int(rng.integers(3, 9)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    object_width, object_height = int(width * object_fraction), int(height * object_fraction)
    texture = rng.integers(0, 255, (max(1, object_height // 6), max(1, object_width // 6), 3), dtype=np.uint8)
    block = cv2.resize(texture, (object_width, object_height), interpolation=cv2.INTER_NEAREST)
    camera = 100.0 + rng.normal(0.0, jitter, (count, 2))
    camera[0] = 100.0
    camera[:, 0] += np.linspace(0.0, pan, count)
    frames = []
    for index, (x, y) in enumerate(camera):
        matrix = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]], dtype=np.float32)
        frame = cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR)
        left = int(width * 0.15 + OBJECT_SPEED_PX * index) % max(1, width - object_width)
        top = int(height * 0.25)
        frame[top : top + object_height, left : left + object_width] = block
        frames.append(frame)
    return frames, np.diff(camera, axis=0)


def report_object_tracking(frames: list[np.ndarray], true_deltas: np.ndarray) -> None:
    """How often each auto mode's inter-frame estimate matched the camera (and how often the block)."""
    for mode in (StabilizationMode.AUTO, StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE):
        state = StabilizationState()
        camera_ok = followed_block = estimates = 0
        for index, frame in enumerate(frames):
            transform = estimators.estimate_inter_frame_transform(frame, index, state, mode)
            if index == 0 or transform is None:
                continue
            estimates += 1
            error = np.array([transform[0, 2], transform[1, 2]]) - true_deltas[index - 1]
            camera_ok += float(np.linalg.norm(error)) < 1.5
            followed_block += abs(error[0] + OBJECT_SPEED_PX) < 2.0
        print(
            f"{mode.value:12s} estimativas {estimates:3d}/{len(frames) - 1}  "
            f"camera (erro < 1,5 px) {camera_ok:3d}  seguiu o bloco {followed_block:3d}"
        )


def positions_of(frames: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shifts = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _response = cv2.phaseCorrelate(reference_gray, gray)
        shifts.append((dx, dy))
    return np.array(shifts)


def report(label: str, frames: list[np.ndarray], reference: np.ndarray) -> None:
    positions = positions_of(frames, reference)
    steps = np.diff(positions, axis=0)
    count = len(positions)
    trend = np.stack([np.linspace(positions[0, axis], positions[-1, axis], count) for axis in range(2)], axis=1)
    detrended = positions - trend
    print(f"{label:52s} passo a passo std={np.std(steps):5.2f} px | tremor std={np.std(detrended):5.2f} px")


def run(
    frames: list[np.ndarray],
    mode: StabilizationMode,
    intensity: float,
    smoothing: float,
    reference: np.ndarray | None = None,
) -> list[np.ndarray]:
    settings = StabilizationSettings(
        mode=mode, intensity=intensity, temporal_smoothing=smoothing, focus_rect=FOCUS, auto_crop=False
    )
    state = StabilizationState()
    return [
        apply_stabilization(frame, settings, frame_index=index, state=state, reference_frame=reference)
        for index, frame in enumerate(frames)
    ]


def offline_reference(frames: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Upper bound: full trajectory known in advance, smoothed with a centred moving average."""
    state = StabilizationState()
    motions = [np.zeros(2)]
    for index, frame in enumerate(frames):
        shift = estimators.estimate_translation_shift(frame, index, state)
        if index > 0:
            motions.append(np.array(shift) if shift is not None else np.zeros(2))
    path = np.cumsum(-np.array(motions), axis=0)
    kernel = np.ones(window) / window
    smooth = np.stack(
        [
            np.convolve(np.pad(path[:, axis], (window // 2, window // 2), mode="edge"), kernel, mode="valid")[
                : len(path)
            ]
            for axis in range(2)
        ],
        axis=1,
    )
    correction = smooth - path
    height, width = frames[0].shape[:2]
    return [
        cv2.warpAffine(
            frame,
            np.array([[1.0, 0.0, correction[index, 0]], [0.0, 1.0, correction[index, 1]]], dtype=np.float32),
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )
        for index, frame in enumerate(frames)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--jitter", type=float, default=6.0, help="desvio padrao do tremor em px")
    parser.add_argument("--pan", type=float, default=20.0, help="panoramica total em px")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--object", type=float, default=0.0, help="fracao do frame ocupada por um bloco texturizado em movimento"
    )
    args = parser.parse_args()

    if args.object > 0:
        frames, true_deltas = make_object_sequence(args.frames, args.jitter, args.pan, args.seed, args.object)
        print(f"Bloco em movimento cobrindo {args.object:.0%} do frame sobre fundo liso:")
        report_object_tracking(frames, true_deltas)
        return  # the phase-correlation shake report would measure the block, not the camera

    frames = make_sequence(args.frames, args.jitter, args.pan, args.seed)
    reference = frames[0]
    report(f"ENTRADA (tremor std {args.jitter:g} px + pan {args.pan:g} px)", frames, reference)
    for mode in (StabilizationMode.AUTO, StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE):
        for intensity, smoothing in ((1.0, 0.0), (1.0, 0.35), (0.65, 0.35), (1.0, 0.8)):
            label = f"{mode.value:12s} intensidade={intensity:.2f} suavizacao={smoothing:.2f}"
            report(label, run(frames, mode, intensity, smoothing), reference)
    report("camera_lock  intensidade=1.00", run(frames, StabilizationMode.CAMERA_LOCK, 1.0, 0.0, reference), reference)
    report(
        "hybrid       intensidade=1.00 suavizacao=0.35",
        run(frames, StabilizationMode.HYBRID, 1.0, 0.35, reference),
        reference,
    )
    report("REFERENCIA offline: trajetoria + media movel 15", offline_reference(frames, 15), reference)


if __name__ == "__main__":
    main()
