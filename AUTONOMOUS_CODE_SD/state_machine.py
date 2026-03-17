"""
Centralized state machine and navigation helpers for the autonomous course robot.

How this module fits into the system:
- `course_autonomous_depth.py` owns sensors/inference (RealSense + Hailo YOLO), then calls
    handlers in this file once per frame based on `user_data.mode`.
- This module turns detections + depth into motion/arm commands for the Pico firmware.
- All runtime knobs (speed bands, thresholds, frame counters, search behavior, etc.) come from
    `autonomy_tuning.py` so field tuning can happen without rewriting state logic.

Primary data contracts used here:
- `detections`: list of `(label, confidence, bbox)` produced by Hailo post-process.
- `bbox`: normalized box object exposing `xmin/xmax/ymin/ymax` in [0, 1].
- `ud` (`user_data`): shared mutable runtime state owned by caller.
- Pico serial command format: `lin_vel, ang_vel, shoulder_cmd, claw_cmd, arm_state\n`.
"""

from __future__ import annotations

import re

import autonomy_tuning as tune

# -----------------------------------------------------------------------------
# Constants: imported from autonomy_tuning.py so this file stays logic-only.
# -----------------------------------------------------------------------------
# Real-world object heights used by geometry fallback distance estimation.
BALL_HEIGHT_M = tune.BALL_HEIGHT_M
BUCKET_HEIGHT_M = tune.BUCKET_HEIGHT_M

# Effective focal length in pixels used by Z = (f * H) / h.
FOCAL_PX = tune.FOCAL_PX

# Detection confidence and temporal filtering settings.
MIN_CONFIDENCE = tune.MIN_CONFIDENCE
CONFIRM_FRAMES = tune.CONFIRM_FRAMES
CLOSE_CONFIRM_FRAMES = tune.CLOSE_CONFIRM_FRAMES

# Pico message format: "lin_vel, ang_vel, shoulder_cmd, claw_cmd, arm_state\n"
# arm_state: 0 = idle, 10 = neutral/return
def build_msg(lin_vel: float, ang_vel: float, shoulder: int, claw: int, arm_state: int = 0) -> bytes:
    """
    Build the 5-field ASCII command consumed by Pico firmware.

    Field origins:
    - `lin_vel`, `ang_vel`: navigation control outputs from this state machine.
    - `shoulder`, `claw`: arm motor command magnitudes (signed setpoints expected by Pico code).
    - `arm_state`: lightweight mode flag used by lower-level arm logic (0=idle, 10=neutral/return).
    """
    return f"{lin_vel:.2f}, {ang_vel:.2f}, {shoulder}, {claw}, {arm_state}\n".encode("utf-8")


# -----------------------------------------------------------------------------
# Best detection: filter by class and pick one (by confidence, then center distance)
# -----------------------------------------------------------------------------
def _bbox_center(bbox) -> tuple[float, float]:
    """Return normalized bbox center (cx, cy) in [0,1] from Hailo box coordinates."""
    cx = (bbox.xmin() + bbox.xmax()) / 2.0
    cy = (bbox.ymin() + bbox.ymax()) / 2.0
    return cx, cy


def _label_tokens(label: str) -> list[str]:
    """Split a model class label into lowercase alphanumeric tokens."""
    return [token for token in re.split(r"[^a-z0-9]+", label.lower()) if token]


def _base_color_key(text: str) -> str:
    """Normalize class-like strings to a color-family key (e.g. red_ball -> red)."""
    key = text.lower().strip()
    for suffix in ("_ball", "_bucket"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    key = key.replace("ball", " ").replace("bucket", " ")
    key = "_".join(token for token in _label_tokens(key))
    return key


def _known_base_colors() -> set[str]:
    """Return normalized base color names derived from `tune.KNOWN_COLOR_TOKENS`."""
    return {_base_color_key(token) for token in tune.KNOWN_COLOR_TOKENS if _base_color_key(token)}


def _extract_color_from_label(label: str) -> str | None:
    """
    Extract color family from a class label (e.g. red_ball/red_bucket -> red).

    Resolution order:
    1) Direct match against configured known color tokens in `autonomy_tuning.py`.
    2) Tokenized fallback for labels with separators/alternate ordering.
    """
    lower_label = label.lower()
    known_bases = _known_base_colors()
    for class_token in tune.KNOWN_COLOR_TOKENS:
        token_l = class_token.lower()
        if token_l in lower_label:
            return _base_color_key(token_l)
    label_tokens = _label_tokens(lower_label)
    for token in label_tokens:
        if token in ("ball", "bucket"):
            continue
        if token in known_bases:
            return token
    return None


def _matches_required_color(label: str, required_color: str) -> bool:
    """
    Match required color robustly across label styles:
    - required can be "red", "red_ball", or "red_bucket"
    - label can be "red_ball", "ball_red", compact, etc.
    """
    req_key = _base_color_key(required_color)
    if not req_key:
        return False
    label_key = _extract_color_from_label(label)
    return label_key == req_key


def _get_requested_ball_color(ud) -> str | None:
    """
    Return the ball color requested by tuning.
    - If TARGET_BALL_COLOR is empty, no color preference is enforced.
    - If TARGET_ONLY_ON_FIRST_LAP is True, enforce only while lap_counter == 0.
    """
    # Optional mission preference configured in autonomy_tuning.py.
    requested = str(getattr(tune, "TARGET_BALL_COLOR", "")).strip().lower()
    if not requested:
        return None
    first_lap_only = bool(getattr(tune, "TARGET_ONLY_ON_FIRST_LAP", True))
    if first_lap_only and getattr(ud, "lap_counter", 0) > 0:
        return None
    return requested


def pick_best_detection(
    detections,
    class_substring: str,
    min_conf: float = MIN_CONFIDENCE,
    required_color: str | None = None,
):
    """
    Pick one target from Hailo detections.

    Input source:
    - `detections` is produced by `HailoInference.get_latest()` in course_autonomous_depth.py.

    Selection rule:
    1) Keep class matches (`ball` or `bucket`) with confidence >= `min_conf`.
    2) Optionally enforce color (`required_color`) using robust label normalization.
    3) Choose highest confidence; break ties by closest-to-center.

    Returns:
    - `(label, conf, bbox)` for best candidate, or `None` if no valid candidate.
    """
    class_substring = class_substring.lower().strip()
    required_color = required_color.lower().strip() if required_color else None
    candidates = []
    for (label, conf, bbox) in detections:
        lower_label = label.lower()
        if class_substring not in lower_label or conf < min_conf:
            continue
        if required_color:
            if not _matches_required_color(label, required_color):
                continue
        candidates.append((label, conf, bbox))
    if not candidates:
        return None
    # Sort by confidence descending, then by distance from center ascending.
    # We use min() because score is (-conf, dist). Lower tuple means better candidate.
    def score(item):
        label, conf, bbox = item
        cx, cy = _bbox_center(bbox)
        dist_from_center = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
        return (-conf, dist_from_center)
    best = min(candidates, key=score)
    return best


# -----------------------------------------------------------------------------
# Distance: depth-first, then geometry fallback
# -----------------------------------------------------------------------------
def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def _turn_rate_from_center(center_x: float) -> float:
    """
    Convert horizontal pixel error into angular velocity command.

    Gains/limits come from `autonomy_tuning.py`:
    - CENTER_DEADBAND: ignore tiny jitter near center.
    - STEER_KP: proportional gain from normalized error to turn-rate.
    - MAX_TURN_RATE: hard clamp for safety/stability.
    """
    err = center_x - 0.5
    if abs(err) < tune.CENTER_DEADBAND:
        return 0.0
    return _clamp(tune.STEER_KP * err, -tune.MAX_TURN_RATE, tune.MAX_TURN_RATE)


def _search_motion(lost_streak: int, base_turn_rate: float, forward_lin: float, preferred_dir: float) -> tuple[float, float]:
    """
    Generate a phased search command when a target is missing:
    1) narrow sweep in last-seen direction
    2) wide sweep opposite direction
    3) forward arc in first direction
    4) forward arc opposite direction
    """
    # All timings/multipliers are tuneable so search can be adapted to course layout.
    phase_frames = max(1, int(getattr(tune, "SEARCH_PHASE_FRAMES", 24)))
    active_lost = max(0, lost_streak - tune.SEARCH_START_FRAMES)
    phase = (active_lost // phase_frames) % 4
    wide_mul = float(getattr(tune, "SEARCH_WIDE_TURN_MULT", 1.45))
    arc_turn_mul = float(getattr(tune, "SEARCH_ARC_TURN_MULT", 0.55))
    dir_sign = -1.0 if preferred_dir < 0 else 1.0
    if phase == 0:
        return (0.0, dir_sign * base_turn_rate)
    if phase == 1:
        return (0.0, -dir_sign * base_turn_rate * wide_mul)
    if phase == 2:
        return (forward_lin, dir_sign * base_turn_rate * arc_turn_mul)
    return (forward_lin, -dir_sign * base_turn_rate * arc_turn_mul)


def _depth_median_at_center(depth_frame, cx_px: int, cy_px: int, width: int, height: int) -> float:
    """
    Compute median depth from a small kernel around target center.
    This is more robust than a single pixel.
    """
    if depth_frame is None:
        return 0.0
    # Kernel size comes from tuning and trades noise rejection vs. spatial specificity.
    radius = max(0, int(tune.DEPTH_KERNEL_RADIUS))
    dvals = []
    for y in range(cy_px - radius, cy_px + radius + 1):
        if y < 0 or y >= height:
            continue
        for x in range(cx_px - radius, cx_px + radius + 1):
            if x < 0 or x >= width:
                continue
            try:
                d = depth_frame.get_distance(x, y)
            except Exception:
                d = 0.0
            if d > tune.MIN_VALID_DEPTH_M:
                dvals.append(d)
    if not dvals:
        return 0.0
    dvals.sort()
    return dvals[len(dvals) // 2]


def compute_distance(
    depth_frame,
    depth_width: int,
    depth_height: int,
    bbox,
    model_height: int,
    real_height_m: float,
) -> tuple[float, bool]:
    """
    Compute target distance in meters using depth-first fusion.

    Input origins:
    - `depth_frame` is aligned RealSense depth from course_autonomous_depth.py.
    - `bbox` is normalized detector output from Hailo YOLO on a 640x640 model input.
    - `real_height_m` is a tuned physical prior from autonomy_tuning.py.

    Method:
    1) Sample robust median depth around bbox center in depth image coordinates.
    2) Compute geometric estimate from apparent bbox height: Z = (FOCAL_PX * H) / h.
    3) Prefer hardware depth when valid, otherwise use geometry fallback.

    Returns:
    - `(distance_m, used_depth)` where `used_depth` marks the chosen source.
    """
    cx_norm = (bbox.xmin() + bbox.xmax()) / 2.0
    cy_norm = (bbox.ymin() + bbox.ymax()) / 2.0
    # Map normalized detector coordinates into aligned depth pixel coordinates.
    cx_px = int(cx_norm * depth_width)
    cy_px = int(cy_norm * depth_height)
    cx_px = max(0, min(depth_width - 1, cx_px))
    cy_px = max(0, min(depth_height - 1, cy_px))

    hw_dist = _depth_median_at_center(depth_frame, cx_px, cy_px, depth_width, depth_height)

    # Geometry fallback from pinhole-camera approximation.
    h_pixels = (bbox.ymax() - bbox.ymin()) * model_height
    if h_pixels < 1.0:
        h_pixels = 1.0
    calc_dist = (FOCAL_PX * real_height_m) / h_pixels

    if hw_dist > tune.MIN_VALID_DEPTH_M:
        return (hw_dist, True)
    return (calc_dist, False)


# -----------------------------------------------------------------------------
# Steering helper: legacy/simple banded logic.
# Kept for compatibility/reference; detect handlers below use finer tune-driven logic.
# -----------------------------------------------------------------------------
def steering_from_bbox_and_distance(
    bbox,
    distance_m: float,
    is_ball: bool,
    forward_positive: bool,
) -> tuple[float, float]:
    """
    Returns (lin_vel, ang_vel) to approach the target.
    - is_ball: True for ball (drive forward = negative lin_vel in current convention), False for bucket.
    - forward_positive: if True, positive lin_vel = drive toward target; if False, negative = toward target.
    """
    center_x = (bbox.xmin() + bbox.xmax()) / 2.0
    # Steer: left of center (center_x < 0.5) -> turn left (negative ang for some conventions)
    # We use: center_x < 0.4 -> turn left, center_x > 0.6 -> turn right, else straight
    if center_x < 0.4:
        ang = 0.5
    elif center_x > 0.6:
        ang = -0.5
    else:
        ang = 0.0

    if is_ball:
        # Toward ball: user_data convention was negative lin_vel
        if distance_m >= tune.BALL_FAR_M:
            lin = -0.35 if forward_positive else -0.35
            ang = -ang  # match course_camera sign
        elif distance_m >= tune.BALL_MID_M:
            lin = -0.2
            ang = -ang
        else:
            lin = -0.2
            ang = -ang
        if not forward_positive:
            lin = -lin
        return (lin, ang)
    else:
        # Toward bucket
        if distance_m >= tune.BUCKET_FAR_M:
            lin = -0.2
            ang = -ang
        elif distance_m >= tune.BUCKET_MID_M:
            lin = -0.1
            ang = -ang
        else:
            lin = -0.1
            ang = -ang
        if not forward_positive:
            lin = -lin
        return (lin, ang)


# -----------------------------------------------------------------------------
# Mode handlers: each takes user_data and optional detections/depth; updates user_data
# -----------------------------------------------------------------------------
def handle_pause(ud):
    """Stop all motion."""
    ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_pick(ud) -> None:
    """Run arm sequence: lower -> close -> raise, then transition to fixed_back."""
    # These frame counts come from tuning and represent timed phases of the pick sequence.
    pick_lower_frames = max(1, int(getattr(tune, "PICK_LOWER_FRAMES", 170)))
    pick_close_frames = max(1, int(getattr(tune, "PICK_CLOSE_FRAMES", 150)))
    pick_raise_frames = max(1, int(getattr(tune, "PICK_RAISE_FRAMES", 180)))
    if ud.arm_state == "idle":
        ud.arm_state = "lower"
        ud.picker_counter = 0
    ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
    if ud.arm_state == "lower":
        # Shoulder down command amplitude is firmware-specific (expected by Pico control loop).
        ud.latest_msg = build_msg(0.0, 0.0, 3000, 0, 0)
        ud.picker_counter += 1
        if ud.picker_counter >= pick_lower_frames:
            ud.arm_state = "close"
            ud.picker_counter = 0
    elif ud.arm_state == "close":
        # Close claw while holding shoulder command neutral.
        ud.latest_msg = build_msg(0.0, 0.0, 0, 3000, 0)
        ud.picker_counter += 1
        if ud.picker_counter >= pick_close_frames:
            ud.arm_state = "raise"
            ud.picker_counter = 0
    elif ud.arm_state == "raise":
        # Shoulder up command (negative direction by current motor convention).
        ud.latest_msg = build_msg(0.0, 0.0, -3000, 0, 0)
        ud.picker_counter += 1
        if ud.picker_counter >= pick_raise_frames:
            ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
            ud.mode = "fixed_back"
            ud.picker_counter = 0
            ud.arm_state = "idle"


def handle_drop(ud) -> None:
    """Lower arm, open claw, then go to detect (next ball) or swivel_large_right (done)."""
    # Like pick, drop timing is fully tune-controlled.
    drop_lower_frames = max(1, int(getattr(tune, "DROP_LOWER_FRAMES", 40)))
    drop_open_frames = max(1, int(getattr(tune, "DROP_OPEN_FRAMES", 40)))
    turn_to_center_after_drop = bool(getattr(tune, "TURN_TO_CENTER_AFTER_DROP", True))
    ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
    if ud.arm_state == "lower":
        ud.latest_msg = build_msg(0.0, 0.0, 3000, 0, 0)
        ud.picker_counter += 1
        if ud.picker_counter >= drop_lower_frames:
            ud.arm_state = "open"
            ud.picker_counter = 0
    elif ud.arm_state == "open":
        ud.latest_msg = build_msg(0.0, 0.0, 0, -3000, 0)
        ud.picker_counter += 1
        if ud.picker_counter >= drop_open_frames:
            ud.lap_counter += 1
            ud.carrying_ball_color = None
            ud.target_bucket_color = None
            if turn_to_center_after_drop:
                ud.mode = "swivel_large_right"
            else:
                ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 10)
                ud.mode = "detect"
                ud.picker_counter = 0
                ud.fixed_travel_counter = 0
            ud.arm_state = "idle"


def handle_fixed_ball(ud) -> None:
    """Drive forward (encoder phase) toward center; then switch to detect."""
    # Open-loop timed travel; useful when no target is visible yet.
    fixed_ball_frames = max(1, int(getattr(tune, "FIXED_BALL_TRAVEL_FRAMES", 500)))
    ud.latest_msg = build_msg(-0.30, 0.0, 0, 0, 10)
    ud.fixed_travel_counter += 1
    if ud.fixed_travel_counter >= fixed_ball_frames:
        ud.mode = "detect"
        ud.fixed_travel_counter = 0
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_fixed_bucket(ud) -> None:
    """Drive forward toward bucket area; then switch to detect_bucket."""
    # Open-loop transition leg between pick and bucket-search phases.
    fixed_bucket_frames = max(1, int(getattr(tune, "FIXED_BUCKET_TRAVEL_FRAMES", 300)))
    ud.latest_msg = build_msg(-0.30, 0.0, 0, 0, 0)
    ud.fixed_travel_counter += 1
    if ud.fixed_travel_counter >= fixed_bucket_frames:
        ud.mode = "detect_bucket"
        ud.fixed_travel_counter = 0
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_fixed_back(ud) -> None:
    """Back up a bit after pick; then swivel_small_left."""
    fixed_back_frames = max(1, int(getattr(tune, "FIXED_BACK_TRAVEL_FRAMES", 80)))
    ud.latest_msg = build_msg(0.1, 0.0, 0, 0, 0)
    ud.fixed_travel_counter += 1
    if ud.fixed_travel_counter >= fixed_back_frames:
        ud.mode = "swivel_small_left"
        ud.fixed_travel_counter = 0
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_swivel_small_left(ud) -> None:
    """Turn left ~45 deg; then fixed_bucket."""
    swivel_small_left_frames = max(1, int(getattr(tune, "SWIVEL_SMALL_LEFT_FRAMES", 95)))
    ud.latest_msg = build_msg(0.0, -0.4, 0, 0, 0)
    ud.fixed_travel_counter += 1
    if ud.fixed_travel_counter >= swivel_small_left_frames:
        ud.mode = "fixed_bucket"
        ud.fixed_travel_counter = 0
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_swivel_large_right(ud) -> None:
    """Turn right ~180 deg back toward center; then fixed_ball."""
    swivel_large_right_frames = max(1, int(getattr(tune, "SWIVEL_LARGE_RIGHT_FRAMES", 540)))
    ud.latest_msg = build_msg(0.0, 0.4, 0, 0, 0)
    ud.fixed_travel_counter += 1
    if ud.fixed_travel_counter >= swivel_large_right_frames:
        ud.mode = "fixed_ball"
        ud.fixed_travel_counter = 0
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)


def handle_detect_ball(
    ud,
    detections: list,
    depth_frame,
    depth_width: int,
    depth_height: int,
    model_height: int = 640,
) -> None:
    """
    Ball approach state.

    Called each frame from course_autonomous_depth.py when `ud.mode == "detect"`.
    It fuses perception + temporal confirmation + proximity gating to decide whether to:
    - search,
    - drive toward target,
    - back off if too close,
    - or transition to `pick`.
    """
    seen_streak = getattr(ud, "ball_seen_streak", 0)
    close_streak = getattr(ud, "ball_close_streak", 0)
    lost_streak = getattr(ud, "ball_lost_streak", 0)

    # Requested color policy is optional and controlled from tuning file.
    requested_ball_color = _get_requested_ball_color(ud)
    best = pick_best_detection(
        detections,
        "ball",
        MIN_CONFIDENCE,
        required_color=requested_ball_color,
    )
    if requested_ball_color and best is None:
        # If requested color is missing for long enough, optional fallback can widen to any ball.
        target_missing = getattr(ud, "ball_target_missing_streak", 0) + 1
        ud.ball_target_missing_streak = target_missing
        allow_fallback = bool(getattr(tune, "FALLBACK_TO_ANY_BALL_IF_TARGET_MISSING", True))
        fallback_frames = max(1, int(getattr(tune, "TARGET_FALLBACK_FRAMES", 45)))
        if allow_fallback and target_missing >= fallback_frames:
            best = pick_best_detection(detections, "ball", MIN_CONFIDENCE)
    else:
        ud.ball_target_missing_streak = 0

    if best is None:
        # Target lost: ramp into an active search pattern after configurable grace period.
        ud.ball_seen_streak = 0
        ud.ball_close_streak = 0
        lost_streak += 1
        ud.ball_lost_streak = lost_streak
        if lost_streak >= tune.SEARCH_START_FRAMES:
            search_dir = getattr(ud, "ball_search_dir", 1.0)
            lin_cmd, ang_cmd = _search_motion(
                lost_streak,
                tune.BALL_SEARCH_TURN_RATE,
                tune.BALL_SEARCH_FORWARD_LIN,
                search_dir,
            )
            ud.latest_msg = build_msg(lin_cmd, ang_cmd, 0, 0, 0)
        else:
            ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
        return

    ud.ball_lost_streak = 0
    # Require N stable frames before trusting target, reducing one-frame false positives.
    seen_streak += 1
    ud.ball_seen_streak = seen_streak
    if seen_streak < CONFIRM_FRAMES:
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
        return

    label, conf, bbox = best
    detected_ball_color = _extract_color_from_label(label)
    if detected_ball_color is None and requested_ball_color:
        # If label taxonomy is color-implicit, preserve configured target color.
        detected_ball_color = requested_ball_color
    dist_m, used_depth = compute_distance(
        depth_frame, depth_width, depth_height, bbox, model_height, BALL_HEIGHT_M
    )
    center_x = (bbox.xmin() + bbox.xmax()) / 2.0
    ud.ball_search_dir = -1.0 if center_x < 0.5 else 1.0
    ang_cmd = _turn_rate_from_center(center_x)

    ud.distance = dist_m
    ud.distance_from_depth = used_depth

    # Farther than pick window: approach with speed bands and steering.
    if dist_m > tune.BALL_PICK_MAX_M:
        ud.ball_close_streak = 0
        if dist_m >= tune.BALL_FAR_M:
            lin_cmd = tune.BALL_SPEED_FAR
        elif dist_m >= tune.BALL_MID_M:
            lin_cmd = tune.BALL_SPEED_MID
        else:
            lin_cmd = tune.BALL_SPEED_NEAR
        if abs(center_x - 0.5) > tune.MISALIGN_ERR_FOR_SLOWDOWN:
            lin_cmd *= tune.MISALIGN_LINEAR_SCALE
        ud.latest_msg = build_msg(lin_cmd, ang_cmd, 0, 0, 0)
        return

    # Inside too-close zone: back up slightly while still trying to steer on target.
    if dist_m < tune.BALL_PICK_MIN_M:
        ud.ball_close_streak = 0
        ud.latest_msg = build_msg(tune.BALL_TOO_CLOSE_BACKUP_LIN, ang_cmd * 0.5, 0, 0, 0)
        return

    # In the pick distance window, require center alignment for several frames.
    if abs(center_x - 0.5) <= tune.BALL_CENTER_TOL_FOR_PICK:
        close_streak += 1
    else:
        close_streak = 0
    ud.ball_close_streak = close_streak

    if close_streak < CLOSE_CONFIRM_FRAMES:
        ud.latest_msg = build_msg(0.0, ang_cmd, 0, 0, 0)
        return

    # Commit to pick sequence and remember inferred carried-ball color for bucket targeting.
    ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
    ud.arm_state = "lower"
    ud.mode = "pick"
    ud.carrying_ball_color = detected_ball_color
    ud.target_bucket_color = detected_ball_color
    ud.ball_seen_streak = 0
    ud.ball_close_streak = 0
    ud.ball_lost_streak = 0


def handle_detect_bucket(
    ud,
    detections: list,
    depth_frame,
    depth_width: int,
    depth_height: int,
    model_height: int = 640,
) -> None:
    """
    Bucket approach state.

    Mirrors ball logic but uses bucket thresholds and transitions to `drop` when aligned/close.
    If `ud.target_bucket_color` is set after pick, matching can be color-constrained.
    """
    seen_streak = getattr(ud, "bucket_seen_streak", 0)
    close_streak = getattr(ud, "bucket_close_streak", 0)
    lost_streak = getattr(ud, "bucket_lost_streak", 0)

    # Set during successful pick so drop can target matching bucket color.
    target_bucket_color = getattr(ud, "target_bucket_color", None)
    best = pick_best_detection(
        detections,
        "bucket",
        MIN_CONFIDENCE,
        required_color=target_bucket_color,
    )
    if target_bucket_color and best is None:
        # Optional widening to any bucket after missing desired color long enough.
        target_missing = getattr(ud, "bucket_target_missing_streak", 0) + 1
        ud.bucket_target_missing_streak = target_missing
        allow_fallback = bool(getattr(tune, "BUCKET_FALLBACK_TO_ANY_IF_TARGET_MISSING", False))
        fallback_frames = max(1, int(getattr(tune, "BUCKET_TARGET_FALLBACK_FRAMES", 120)))
        if allow_fallback and target_missing >= fallback_frames:
            best = pick_best_detection(detections, "bucket", MIN_CONFIDENCE)
    else:
        ud.bucket_target_missing_streak = 0

    if best is None:
        # Missing target: eventually switch from hold to active scan pattern.
        ud.bucket_seen_streak = 0
        ud.bucket_close_streak = 0
        lost_streak += 1
        ud.bucket_lost_streak = lost_streak
        if lost_streak >= tune.SEARCH_START_FRAMES:
            search_dir = getattr(ud, "bucket_search_dir", 1.0)
            lin_cmd, ang_cmd = _search_motion(
                lost_streak,
                tune.BUCKET_SEARCH_TURN_RATE,
                tune.BUCKET_SEARCH_FORWARD_LIN,
                search_dir,
            )
            ud.latest_msg = build_msg(lin_cmd, ang_cmd, 0, 0, 0)
        else:
            ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
        return

    ud.bucket_lost_streak = 0
    # Debounce detection to reduce false triggers.
    seen_streak += 1
    ud.bucket_seen_streak = seen_streak
    if seen_streak < CONFIRM_FRAMES:
        ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
        return

    label, conf, bbox = best
    dist_m, used_depth = compute_distance(
        depth_frame, depth_width, depth_height, bbox, model_height, BUCKET_HEIGHT_M
    )
    center_x = (bbox.xmin() + bbox.xmax()) / 2.0
    ud.bucket_search_dir = -1.0 if center_x < 0.5 else 1.0
    ang_cmd = _turn_rate_from_center(center_x)

    ud.distance = dist_m
    ud.distance_from_depth = used_depth

    # Too far to drop: approach with tuned speed bands.
    if dist_m > tune.BUCKET_DROP_MAX_M:
        ud.bucket_close_streak = 0
        if dist_m >= tune.BUCKET_FAR_M:
            lin_cmd = tune.BUCKET_SPEED_FAR
        elif dist_m >= tune.BUCKET_MID_M:
            lin_cmd = tune.BUCKET_SPEED_MID
        else:
            lin_cmd = tune.BUCKET_SPEED_NEAR
        if abs(center_x - 0.5) > tune.MISALIGN_ERR_FOR_SLOWDOWN:
            lin_cmd *= tune.MISALIGN_LINEAR_SCALE
        ud.latest_msg = build_msg(lin_cmd, ang_cmd, 0, 0, 0)
        return

    # Too close: small backup to re-enter drop window.
    if dist_m < tune.BUCKET_DROP_MIN_M:
        ud.bucket_close_streak = 0
        ud.latest_msg = build_msg(tune.BUCKET_TOO_CLOSE_BACKUP_LIN, ang_cmd * 0.5, 0, 0, 0)
        return

    # Require center alignment to avoid dropping beside the bucket.
    if abs(center_x - 0.5) <= tune.BUCKET_CENTER_TOL_FOR_DROP:
        close_streak += 1
    else:
        close_streak = 0
    ud.bucket_close_streak = close_streak

    if close_streak < CLOSE_CONFIRM_FRAMES:
        ud.latest_msg = build_msg(0.0, ang_cmd, 0, 0, 0)
        return

    ud.latest_msg = build_msg(0.0, 0.0, 0, 0, 0)
    ud.arm_state = "lower"
    ud.mode = "drop"
    ud.bucket_seen_streak = 0
    ud.bucket_close_streak = 0
    ud.bucket_lost_streak = 0
