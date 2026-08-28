"""Deinterlacing: undo the comb that interlaced capture leaves on anything that moved.

An interlaced frame is two *fields* captured a half frame period apart and woven line by line
(even rows one instant, odd rows the next). Wherever the scene moved between them, edges show
as thin alternating horizontal teeth — the "comb". The detector and the filter both rest on one
per-pixel test, from the classic decomb/yadif family: a row is combed where it differs from
*both* the row above and the row below in the same direction, ``(a - b) * (c - b) > T²`` — a
clean step edge gives 0 there, a comb tooth gives a large positive product.

Everything is per frame and stateless, so preview and export agree, and it has to run on the
frame at its native size, before any resize mixes the rows.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.settings import DeinterlaceMode

COMB_THRESHOLD = 24
"""8-bit luma difference a row must show against both neighbours to count as a comb tooth."""
DETECTION_ROW_STRIDE = 8
DETECTION_COLUMN_STRIDE = 4
"""The detector samples one row triple in every eight rows and one column in four: cheap on 4K."""
LIKELY_INTERLACED_SCORE = 0.01
"""Comb fraction above which a frame is reported as probably interlaced (progressive video sits well below)."""
MIN_ROWS = 4


def _luma(frame: np.ndarray) -> np.ndarray:
    """Green channel as luma (no colour conversion), wide enough for the product not to overflow."""
    return (frame[..., 1] if frame.ndim == 3 else frame).astype(np.int32)


def comb_mask(above: np.ndarray, row: np.ndarray, below: np.ndarray, threshold: int = COMB_THRESHOLD) -> np.ndarray:
    """True where ``row`` (luma) sticks out from both neighbours in the same direction."""
    return (above - row) * (below - row) > threshold * threshold


def comb_score(frame: np.ndarray) -> float:
    """Fraction of sampled pixels that are comb teeth: ~0 for progressive video, percents for interlaced motion."""
    height = frame.shape[0]
    if height < MIN_ROWS:
        return 0.0
    luma = _luma(frame)
    rows = np.arange(1, height - 1, DETECTION_ROW_STRIDE)
    columns = slice(None, None, DETECTION_COLUMN_STRIDE)
    mask = comb_mask(luma[rows - 1, columns], luma[rows, columns], luma[rows + 1, columns])
    return float(mask.mean()) if mask.size else 0.0


def likely_interlaced(score: float) -> bool:
    return score >= LIKELY_INTERLACED_SCORE


def deinterlace(frame: np.ndarray, mode: DeinterlaceMode) -> np.ndarray:
    """``frame`` with the odd rows rebuilt from the even field — everywhere (FIELD) or only where combed (ADAPTIVE).

    Keeping the even (top) field means static areas stay pixel-exact in ADAPTIVE mode; in a combed
    area every odd row is replaced by the average of its even neighbours, so the region shows one
    field consistently instead of two interleaved instants.
    """
    if mode is DeinterlaceMode.OFF or frame.shape[0] < MIN_ROWS:
        return frame
    even = frame[0::2]
    odd = frame[1::2]
    count = odd.shape[0]
    above = even[:count]
    below = even[1 : count + 1]
    if below.shape[0] < count:  # an odd number of rows: the last odd row has no even row below it
        below = np.concatenate([below, above[-1:]], axis=0)
    interpolated = ((above.astype(np.uint16) + below.astype(np.uint16) + 1) >> 1).astype(np.uint8)

    result = frame.copy()
    if mode is DeinterlaceMode.FIELD:
        result[1::2] = interpolated
        return result

    mask = comb_mask(_luma(above), _luma(odd), _luma(below)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))  # cover the rim of a combed area too
    combed = mask.astype(bool)
    if frame.ndim == 3:
        combed = combed[..., None]
    result[1::2] = np.where(combed, interpolated, odd)
    return result


def prepare_source(frame: np.ndarray, mode: DeinterlaceMode) -> tuple[np.ndarray, float]:
    """The frame the rest of the pipeline should see, plus the comb score of the frame as captured."""
    return deinterlace(frame, mode), comb_score(frame)
