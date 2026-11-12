"""Perspective (keystone) correction: four points on a real-world rectangle become a rectangle on screen.

Runs after rotation/flip and before the crop, so the crop rectangle drawn on the corrected
preview is the one exported. The points live in the frame before rotation/flip
(:class:`~space_video_enhancer.settings.PerspectiveSettings`); here they are mapped into the
displayed frame, put in canonical order (top-left, top-right, bottom-right, bottom-left) and
turned into a homography.

What the quadrilateral becomes depends on :class:`~space_video_enhancer.settings.PerspectiveFit`:

- ``REAL``: a rectangle with the plane's estimated real proportion, centred where the
  quadrilateral was and with the same area, in a frame of the same size — the rest of the
  picture follows the plane and the uncovered corners turn black (the crop trims them).
- ``STRETCH_*``: the four points become the corners of a 16:9 or 9:16 output, so the plane
  fills it (stretched away from its real proportion when the shapes differ) and the
  correction never shrinks the picture.

The real proportion comes from Zhang & He's whiteboard rectification ("Whiteboard scanning and
image enhancement", 2007): with square pixels and the principal point at the frame centre, the two
vanishing points of the quadrilateral give the focal length, and the focal length gives the
width/height of the rectangle in the world. When an edge pair is parallel on screen (a symmetric
keystone, the common case) the focal length cannot be measured and a typical phone lens is assumed.

A strong correction can put the plane's horizon inside the frame: beyond it a homography wraps the
picture around (the inverse map flips sign), so the output side of that line is painted black.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from space_video_enhancer.pipeline.geometry import normalized_through_geometry
from space_video_enhancer.settings import GeometrySettings, NormalizedPoint, PerspectiveFit, PerspectiveSettings
from space_video_enhancer.telemetry import trace as tracing

ASSUMED_FOCAL_FRACTION = 0.72
"""Focal length assumed when the points cannot measure it, as a fraction of the frame's long side (~26 mm equiv.)."""
FOCAL_RANGE = (0.25, 4.0)
"""A measured focal length outside this range (times the long side) is noise from near-parallel edges."""
PLANE_ASPECT_RANGE = (0.1, 10.0)
MIN_QUAD_AREA_FRACTION = 0.002
"""Smaller quadrilaterals (fraction of the frame area) are a misclick, not a plane."""
MIN_CORNER_SINE = 0.05
"""Corners flatter or sharper than ~3 degrees make the homography explode."""


@dataclass(frozen=True)
class PerspectivePlan:
    """The warp one frame size gets: everything :func:`apply_perspective` needs, plus what the UI reports."""

    homography: np.ndarray
    """3x3, displayed-frame pixels -> output pixels."""
    output_size: tuple[int, int]
    plane_aspect: float
    """Estimated real width/height of the marked plane (reported even when the fit stretches it)."""
    focal_source: str
    """``measured`` (from the vanishing points) or ``assumed`` (typical lens: an edge pair is parallel)."""
    quad: np.ndarray
    """The ordered quadrilateral in displayed-frame pixels."""


def order_quad(points: np.ndarray) -> np.ndarray:
    """``(4, 2)`` points -> top-left, top-right, bottom-right, bottom-left (clockwise on screen, y down).

    Sorted by angle around the centroid, starting at the corner nearest the top-left, so the
    handles can be dragged in any order without crossing the edges.
    """
    quad = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles, kind="stable")]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of an ordered polygon (any number of vertices)."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
    """Convex, large enough and with no degenerate corner (``quad`` ordered, in pixels)."""
    if width <= 0 or height <= 0 or quad_area(quad) < MIN_QUAD_AREA_FRACTION * width * height:
        return False
    for index in range(4):
        incoming = quad[index] - quad[index - 1]
        outgoing = quad[(index + 1) % 4] - quad[index]
        lengths = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if lengths <= 0.0:
            return False
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        if cross / lengths < MIN_CORNER_SINE:  # clockwise on screen: every turn is positive
            return False
    return True


def estimate_plane_aspect(quad: np.ndarray, width: int, height: int) -> tuple[float, str]:
    """Real width/height of the rectangle imaged as ``quad`` (ordered, pixels) and where the focal came from."""
    u0, v0 = width / 2.0, height / 2.0
    homogeneous = np.hstack([quad, np.ones((4, 1))])
    m1, m2, m4, m3 = homogeneous  # Zhang's numbering: m1 top-left, m2 top-right, m3 bottom-left, m4 bottom-right
    diagonal = np.cross(m1, m4)
    k2 = float(np.dot(diagonal, m3) / np.dot(np.cross(m2, m4), m3))
    k3 = float(np.dot(diagonal, m2) / np.dot(np.cross(m3, m4), m2))
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    long_side = float(max(width, height))
    focal: float | None = None
    if abs(n2[2]) > 1e-9 and abs(n3[2]) > 1e-9:
        squared = -(
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
            + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
        ) / (n2[2] * n3[2])
        if squared > 0.0:
            candidate = math.sqrt(squared)
            if FOCAL_RANGE[0] * long_side <= candidate <= FOCAL_RANGE[1] * long_side:
                focal = candidate
    source = "measured"
    if focal is None:
        focal, source = ASSUMED_FOCAL_FRACTION * long_side, "assumed"

    inverse_intrinsics = np.linalg.inv(np.array([[focal, 0.0, u0], [0.0, focal, v0], [0.0, 0.0, 1.0]]))
    width_ray = inverse_intrinsics @ n2
    height_ray = inverse_intrinsics @ n3
    height_norm = float(np.dot(height_ray, height_ray))
    if height_norm <= 0.0:
        return 1.0, source
    aspect = math.sqrt(float(np.dot(width_ray, width_ray)) / height_norm)
    return min(PLANE_ASPECT_RANGE[1], max(PLANE_ASPECT_RANGE[0], aspect)), source


def displayed_quad(settings: PerspectiveSettings, geometry: GeometrySettings, width: int, height: int) -> np.ndarray:
    """The settings' points in pixels of the displayed ``width x height`` frame, ordered."""
    points = np.array([[point.x, point.y] for point in settings.points], dtype=np.float64)
    displayed = normalized_through_geometry(points, geometry)
    return order_quad(displayed * np.array([width, height], dtype=np.float64) - 0.5)


def plan_perspective(
    settings: PerspectiveSettings, geometry: GeometrySettings, width: int, height: int
) -> PerspectivePlan | None:
    """The warp for a displayed frame of ``width x height``; ``None`` when off or the points do not make a plane."""
    if not settings.is_active or width <= 0 or height <= 0:
        return None
    quad = displayed_quad(settings, geometry, width, height)
    if not is_valid_quad(quad, width, height):
        return None
    plane_aspect, focal_source = estimate_plane_aspect(quad, width, height)
    output_width, output_height = settings.output_size(width, height)
    if settings.fit is PerspectiveFit.REAL:
        center = quad.mean(axis=0)
        area = quad_area(quad)
        half_width = math.sqrt(area * plane_aspect) / 2.0
        half_height = math.sqrt(area / plane_aspect) / 2.0
        left, right = center[0] - half_width, center[0] + half_width
        top, bottom = center[1] - half_height, center[1] + half_height
    else:
        left, top = -0.5, -0.5
        right, bottom = output_width - 0.5, output_height - 0.5
    target = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return PerspectivePlan(homography, (output_width, output_height), plane_aspect, focal_source, quad)


def apply_perspective(frame: np.ndarray, settings: PerspectiveSettings, geometry: GeometrySettings) -> np.ndarray:
    """Warp the displayed ``frame`` so the marked plane becomes a rectangle; identity when off or invalid."""
    if not settings.is_active:
        return frame
    height, width = frame.shape[:2]
    plan = plan_perspective(settings, geometry, width, height)
    if plan is None:
        tracing.note("perspective.result", "invalid")
        return frame
    output_width, output_height = plan.output_size
    warped = cv2.warpPerspective(
        frame,
        plan.homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    wrapped = _blank_beyond_horizon(warped, plan.homography, plan.quad.mean(axis=0))
    tracing.note("perspective.result", "applied")
    tracing.note("perspective.fit", settings.fit.value)
    tracing.note("perspective.plane_aspect", round(plan.plane_aspect, 3))
    tracing.note("perspective.focal", plan.focal_source)
    tracing.note("perspective.horizon_px", wrapped)
    return warped


def _blank_beyond_horizon(warped: np.ndarray, homography: np.ndarray, plane_point: np.ndarray) -> int:
    """Paint black the output beyond the plane's horizon (in place); returns roughly how many pixels.

    An output pixel ``p`` samples the source at ``inv(H) @ p``, whose third coordinate is a linear
    function of ``p``: it has the sign of ``H[2] @ q`` for the source point ``q`` it comes from, and
    flips at the horizon. On the far side ``cv2.warpPerspective`` divides by a number of the other
    sign and samples the picture mirrored — a ghost that is not on the plane. ``plane_point`` (a
    source pixel inside the quadrilateral) tells which side is the plane's.
    """
    height, width = warped.shape[:2]
    a, b, c = np.linalg.inv(homography)[2]
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return 0  # affine: no horizon
    sign = 1.0 if float(homography[2] @ np.array([plane_point[0], plane_point[1], 1.0])) > 0 else -1.0
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]) - 0.5
    values = corners @ np.array([a, b]) + c
    if np.all(sign * values > 0):
        return 0
    beyond = _clip_polygon(corners, -sign * a, -sign * b, -sign * c)
    if len(beyond) < 3:
        return 0
    polygon = np.round(beyond).astype(np.int32)
    cv2.fillConvexPoly(warped, polygon, (0,) * (warped.shape[2] if warped.ndim == 3 else 1))
    return round(quad_area(beyond))


def _clip_polygon(polygon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Part of the convex ``polygon`` where ``a x + b y + c >= 0`` (one Sutherland-Hodgman pass)."""
    kept: list[np.ndarray] = []
    count = len(polygon)
    for index in range(count):
        current, following = polygon[index], polygon[(index + 1) % count]
        current_value = a * current[0] + b * current[1] + c
        following_value = a * following[0] + b * following[1] + c
        if current_value >= 0:
            kept.append(current)
        if (current_value >= 0) != (following_value >= 0):
            t = current_value / (current_value - following_value)
            kept.append(current + t * (following - current))
    return np.array(kept, dtype=np.float64).reshape(-1, 2)


def quad_grid_lines(points: np.ndarray, divisions: int = 4) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Inner lines of a ``divisions x divisions`` grid drawn in perspective on the quadrilateral ``points``.

    ``points`` are in any 2D frame (normalized is fine: the projective map commutes with axis scaling).
    The overlay draws them so the user sees which scene lines the plane's grid follows. Empty when
    the points do not form a usable quadrilateral.
    """
    quad = order_quad(points)
    if quad_area(quad) <= 1e-9:
        return []
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(square, quad.astype(np.float32))
    lines = []
    for step in range(1, divisions):
        t = step / divisions
        for start, end in (((t, 0.0), (t, 1.0)), ((0.0, t), (1.0, t))):
            mapped = cv2.perspectiveTransform(np.array([[start, end]], dtype=np.float64), homography)[0]
            lines.append(((float(mapped[0, 0]), float(mapped[0, 1])), (float(mapped[1, 0]), float(mapped[1, 1]))))
    return lines


def points_tuple(points: np.ndarray) -> tuple[NormalizedPoint, ...]:
    """``(4, 2)`` normalized coordinates -> the settings' point tuple, clamped to the frame."""
    clamped = np.clip(np.asarray(points, dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
    return tuple(NormalizedPoint(float(x), float(y)) for x, y in clamped)
