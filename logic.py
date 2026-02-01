"""
Bra Fit Finder - Core Logic
Modular business logic for bra size estimation from MediaPipe pose landmarks.
"""

import math
from dataclasses import dataclass
from typing import Optional

# MediaPipe Pose landmark indices (33 landmarks, same for solutions.pose and tasks.vision)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class LandmarkCoords:
    """Normalised landmark coordinates (x, y, z in 0–1 range)."""

    x: float
    y: float
    z: float


@dataclass
class ExtractedLandmarks:
    """Key body landmarks extracted for bra fitting."""

    shoulder_left: LandmarkCoords
    shoulder_right: LandmarkCoords
    mid_bust: LandmarkCoords
    under_bust: LandmarkCoords
    hip_left: LandmarkCoords
    hip_right: LandmarkCoords


@dataclass
class BraFitResult:
    """Result of the bra fit calculation."""

    volume_estimate: float
    volume_adjusted: float
    recommended_size: str
    landmarks_detected: bool


def _get_landmarks_list(pose_results) -> Optional[list]:
    """
    Normalise pose results to a list of 33 landmarks.
    Supports both legacy (solutions.pose) and Tasks API (PoseLandmarker) formats.
    """
    if pose_results is None or not hasattr(pose_results, "pose_landmarks"):
        return None
    pl = pose_results.pose_landmarks
    if isinstance(pl, list) and len(pl) > 0:
        return pl[0]  # Tasks API: list of poses, each pose is list of landmarks
    if hasattr(pl, "landmark"):
        return list(pl.landmark)  # Legacy solutions.pose format
    return None


def _landmark_from_list(landmarks_list: list, index: int) -> Optional[LandmarkCoords]:
    """Extract a single landmark from a list of MediaPipe landmarks."""
    if landmarks_list is None or index >= len(landmarks_list):
        return None
    lm = landmarks_list[index]
    x = getattr(lm, "x", None)
    y = getattr(lm, "y", None)
    z = getattr(lm, "z", None)
    if x is None or y is None:
        return None
    return LandmarkCoords(x=float(x), y=float(y), z=float(z or 0))


def _midpoint(a: LandmarkCoords, b: LandmarkCoords) -> LandmarkCoords:
    """Return the midpoint between two landmarks."""
    return LandmarkCoords(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
    )


def _interpolate_vertical(
    shoulder_mid: LandmarkCoords,
    hip_mid: LandmarkCoords,
    fraction: float,
) -> LandmarkCoords:
    """
    Interpolate a point between shoulder and hip midpoints.
    fraction=0 is at shoulder, fraction=1 is at hip.
    """
    return LandmarkCoords(
        x=shoulder_mid.x + fraction * (hip_mid.x - shoulder_mid.x),
        y=shoulder_mid.y + fraction * (hip_mid.y - shoulder_mid.y),
        z=shoulder_mid.z + fraction * (hip_mid.z - shoulder_mid.z),
    )


def _distance(a: LandmarkCoords, b: LandmarkCoords) -> float:
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# -----------------------------------------------------------------------------
# Configuration: Data-Driven Anthropometry
# -----------------------------------------------------------------------------

# Vertical interpolation factors (0.0 = Shoulder, 1.0 = Hip).
# DERIVATION:
# - Avg Torso Length (Shoulder to Hip): ~53cm
# - Avg Sternal Notch to Nipple (Post-partum/Ptotic): ~23.5cm -> ~0.44 ratio
# - Avg Nipple to Inframammary Fold: ~7.5cm -> Total ~31cm -> ~0.53 ratio
RATIO_MID_BUST = 0.44  # Nipple Line (Fullest Point)
RATIO_UNDER_BUST = 0.53  # Inframammary Fold (Band Line)

# Width Ratio: Single Breast Width as % of Total Shoulder Width (Bi-acromial)
# Standard medical approx: 13cm breast width / 37cm shoulder width ≈ 0.35
RATIO_BREAST_WIDTH = 0.35


def extract_landmarks(pose_results) -> Optional[ExtractedLandmarks]:
    """
    Extract landmarks using empirically derived vertical ratios.
    """
    landmarks = _get_landmarks_list(pose_results)
    if landmarks is None:
        return None

    shoulder_left = _landmark_from_list(landmarks, LEFT_SHOULDER)
    shoulder_right = _landmark_from_list(landmarks, RIGHT_SHOULDER)
    hip_left = _landmark_from_list(landmarks, LEFT_HIP)
    hip_right = _landmark_from_list(landmarks, RIGHT_HIP)

    if not all([shoulder_left, shoulder_right, hip_left, hip_right]):
        return None

    shoulder_mid = _midpoint(shoulder_left, shoulder_right)
    hip_mid = _midpoint(hip_left, hip_right)

    # Mid-bust: 44% down (matches ~23-24cm drop from shoulder)
    mid_bust = _interpolate_vertical(shoulder_mid, hip_mid, RATIO_MID_BUST)

    # Under-bust: 53% down (matches ~31cm drop from shoulder)
    under_bust = _interpolate_vertical(shoulder_mid, hip_mid, RATIO_UNDER_BUST)

    return ExtractedLandmarks(
        shoulder_left=shoulder_left,
        shoulder_right=shoulder_right,
        mid_bust=mid_bust,
        under_bust=under_bust,
        hip_left=hip_left,
        hip_right=hip_right,
    )


def calculate_volume_estimate(
    front_landmarks: ExtractedLandmarks,
    side_landmarks: ExtractedLandmarks,
) -> float:
    """
    Calculate volume using Qiao's Hemi-Ellipsoid approximation.
    Ref: Breast Volumetric Analysis (Qiao et al.)
    Formula: V = π/3 * (Width/2) * (Height/2) * Projection
    """
    # 1. Breast Width (Front)
    # Derived from shoulder width using anthropometric ratio (0.35)
    frame_width = _distance(
        front_landmarks.shoulder_left,
        front_landmarks.shoulder_right,
    )
    breast_width = frame_width * RATIO_BREAST_WIDTH

    # 2. Breast Height (Side)
    # Vertical span from Mid-Bust to Under-Bust, doubled (radius to diameter)
    radius_height = _distance(
        side_landmarks.mid_bust,
        side_landmarks.under_bust,
    )
    breast_height = radius_height * 2.0

    # 3. Projection (Side)
    # Distance from mid-spine to nipple (approx) minus rib cage depth.
    # We use the raw side depth scaled down to isolate breast tissue.
    side_depth_raw = _distance(
        side_landmarks.shoulder_left,
        side_landmarks.shoulder_right,
    )
    # Heuristic: Breast projection is roughly 60% of total side profile depth
    breast_projection = side_depth_raw * 0.60

    # 4. Ellipsoid Volume
    # V = π/3 * R_width * R_height * Length
    # Using full diameters: V ≈ 0.52 * W * H * P
    volume_unscaled = 0.52 * breast_width * breast_height * breast_projection

    # Scale factor (calibrated to new unit range)
    scale_factor = 2200.0

    return volume_unscaled * scale_factor


def apply_growth_curve(volume_estimate: float, weeks_postpartum: int) -> float:
    """
    Apply growth curve for early postpartum engorgement.

    If weeks_postpartum < 6, increase volume by 15% to account for
    milk regulation and engorgement.
    """
    if weeks_postpartum < 6:
        return volume_estimate * 1.15
    return volume_estimate


# -----------------------------------------------------------------------------
# Configuration: Recalibrated Hemi-Ellipsoid Mapping
# -----------------------------------------------------------------------------

# Mapping adjusted volume scalars to estimated sizes.
# CALIBRATION NOTE:
# The Hemi-Ellipsoid formula produces lower raw values than the Box method.
# These thresholds are tightened to the 2.0 - 15.0 range typical of this geometry.
VOLUME_SIZE_MAP = {
    2.5: "32A",
    3.5: "32B",
    4.5: "34B",
    5.5: "32C",
    6.5: "34C",
    7.5: "36B",
    8.5: "32D",
    9.5: "34D",
    10.5: "36C",
    11.5: "38B",
    12.5: "34DD",
    13.5: "36D",
    14.5: "38C",
    16.0: "40C",
    18.0: "36DD",
    20.0: "38D",
    float("inf"): "Size Check Required",  # Catch-all for high-variance inputs
}


def volume_to_bra_size(volume_adjusted: float) -> str:
    """
    Map adjusted volume estimate to bra size using the recalibrated thresholds.
    """
    for threshold, size in VOLUME_SIZE_MAP.items():
        if volume_adjusted <= threshold:
            return size

    return "Size Check Required"


def compute_bra_fit(
    front_pose_results,
    side_pose_results,
    weeks_postpartum: int,
) -> BraFitResult:
    """
    Full pipeline: extract landmarks, estimate volume, apply growth curve,
    and map to bra size.
    """
    front_landmarks = extract_landmarks(front_pose_results)
    side_landmarks = extract_landmarks(side_pose_results)

    if front_landmarks is None or side_landmarks is None:
        return BraFitResult(
            volume_estimate=0.0,
            volume_adjusted=0.0,
            recommended_size="",
            landmarks_detected=False,
        )

    volume_estimate = calculate_volume_estimate(front_landmarks, side_landmarks)
    volume_adjusted = apply_growth_curve(volume_estimate, weeks_postpartum)
    recommended_size = volume_to_bra_size(volume_adjusted)

    return BraFitResult(
        volume_estimate=volume_estimate,
        volume_adjusted=volume_adjusted,
        recommended_size=recommended_size,
        landmarks_detected=True,
    )
