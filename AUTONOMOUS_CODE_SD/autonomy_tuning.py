"""
Central tuning values for autonomous behavior.
Edit only this file for quick field tuning.
Units:
- Distance: meters
- Velocity: same units used by Pico command protocol
- Angles: rad/s style command magnitudes
"""

# -----------------------------------------------------------------------------
# Detection Quality Gates
# -----------------------------------------------------------------------------
# Minimum detector confidence accepted as a "real" object.
# Raise this if you get false positives on the carpet/background.
# Lower this if true balls/buckets are being ignored too often.
MIN_CONFIDENCE = 0.35

# Number of consecutive frames a target must be seen before motion is allowed.
# Raise for more stability under flicker; lower for faster response.
CONFIRM_FRAMES = 3

# Number of consecutive close/aligned frames required before pick/drop transition.
# Raise to avoid accidental arm triggers; lower to speed up cycle time.
CLOSE_CONFIRM_FRAMES = 3


# -----------------------------------------------------------------------------
# Color Targeting (ball-first strategy)
# -----------------------------------------------------------------------------
# Ordered list of class/label tokens the parser can recognize in model labels.
# These can be full class names (e.g. red_ball, red_bucket) and the state
# machine will map them to the same color family automatically.
KNOWN_COLOR_TOKENS = (
    "blue ball",
    "blue bucket",
    "green ball",
    "green bucket",
    "red ball",
    "red bucket",
    "yellow ball",
    "yellow bucket",
)

# Requested first ball color/class. Use "" to disable color priority entirely.
# Accepted examples: "red", "red ball", "red_ball", or any value in KNOWN_COLOR_TOKENS.
TARGET_BALL_COLOR = "red ball"

# If True, enforce TARGET_BALL_COLOR only while lap_counter == 0 (first cycle).
# If False, every lap will keep prioritizing TARGET_BALL_COLOR.
TARGET_ONLY_ON_FIRST_LAP = True

# If True, robot can eventually fall back to "any ball" when target color is
# missing for too long. If False, robot will keep searching only that color.
FALLBACK_TO_ANY_BALL_IF_TARGET_MISSING = True

# Frames of missing target color before falling back to any ball.
# Raise for stricter color behavior; lower to keep mission moving.
TARGET_FALLBACK_FRAMES = 45

# For buckets: if False, robot will only approach the color-matched bucket.
# If True, after BUCKET_TARGET_FALLBACK_FRAMES it can use any bucket.
BUCKET_FALLBACK_TO_ANY_IF_TARGET_MISSING = False

# Frames to wait before bucket fallback is allowed (if enabled above).
BUCKET_TARGET_FALLBACK_FRAMES = 120


# -----------------------------------------------------------------------------
# Geometry / Depth Distance Fusion
# -----------------------------------------------------------------------------
# Focal length in pixels for geometry fallback distance equation:
# distance ~= (FOCAL_PX * real_object_height_m) / bbox_height_px
# Re-calibrate if geometry distance is consistently biased.
FOCAL_PX = 3386.0

# Real ball diameter/height used by geometry fallback.
# Update if competition ball size changes.
BALL_HEIGHT_M = 0.1524

# Real bucket height used by geometry fallback.
# Update if bucket model/size changes.
BUCKET_HEIGHT_M = 0.381

# Depth values below this are treated as invalid noise.
# Raise if near-zero spikes appear; lower if camera is very close to objects.
MIN_VALID_DEPTH_M = 0.10

# -----------------------------------------------------------------------------
# State-Machine Timing (frame counts)
# -----------------------------------------------------------------------------
# Initial encoder-only travel toward center before vision takes over.
# Lower this if startup drive feels too long.
FIXED_BALL_TRAVEL_FRAMES = 430

# Travel toward bucket area before bucket detection mode starts.
FIXED_BUCKET_TRAVEL_FRAMES = 300

# Reverse after pickup before rotating toward bucket zone.
FIXED_BACK_TRAVEL_FRAMES = 80

# Small left turn toward bucket zone.
SWIVEL_SMALL_LEFT_FRAMES = 95

# Large right turn to face center/balls again.
SWIVEL_LARGE_RIGHT_FRAMES = 540

# If True, always reorient back toward center immediately after each drop.
# This addresses the "does not turn around after drop" issue.
TURN_TO_CENTER_AFTER_DROP = True

# Pick sequence arm timing.
# Lower PICK_LOWER_FRAMES if the arm drops too low during pickup.
PICK_LOWER_FRAMES = 150
PICK_CLOSE_FRAMES = 150
PICK_RAISE_FRAMES = 180

# Drop sequence arm timing.
DROP_LOWER_FRAMES = 40
DROP_OPEN_FRAMES = 40

# Ball approach and pickup
BALL_FAR_M = 6.0
BALL_MID_M = 1.4

# Lower bound of legal pickup distance window.
# If robot is closer than this, code commands a brief reverse.
BALL_PICK_MIN_M = 0.56
BALL_PICK_MAX_M = 0.67
BALL_TOO_CLOSE_BACKUP_LIN = 0.40

# Linear speed used when ball is farther than BALL_FAR_M.
# Negative means "forward" in your current platform convention.
BALL_SPEED_FAR = -0.35

# Linear speed used in mid band (BALL_MID_M to BALL_FAR_M).
BALL_SPEED_MID = -0.18

# Linear speed used near pickup zone (below BALL_MID_M).
BALL_SPEED_NEAR = -0.10

# Required center alignment tolerance before pick is allowed.
# Smaller value = stricter centering.
BALL_CENTER_TOL_FOR_PICK = 0.08

# Bucket approach and drop
BUCKET_FAR_M = 4.0
BUCKET_MID_M = 1.5

# Lower bound of legal drop distance window.
# If robot is too close, it backs up.
BUCKET_DROP_MIN_M = 0.41

# Upper bound of legal drop distance window.
# If robot is farther than this, it keeps approaching.
BUCKET_DROP_MAX_M = 0.55

# Reverse speed when too close to bucket.
BUCKET_TOO_CLOSE_BACKUP_LIN = 0.40

# Linear speed in far bucket band.
BUCKET_SPEED_FAR = -0.35

# Linear speed in mid bucket band.
BUCKET_SPEED_MID = -0.14

# Linear speed in near bucket band.
BUCKET_SPEED_NEAR = -0.08

# Required center alignment tolerance before drop is allowed.
BUCKET_CENTER_TOL_FOR_DROP = 0.10


# -----------------------------------------------------------------------------
# Steering and Alignment Response
# -----------------------------------------------------------------------------
# Proportional steering gain on horizontal image error.
# Raise for stronger turn response; lower if steering oscillates.
STEER_KP = 1.6

# Steering command hard clamp.
MAX_TURN_RATE = 0.55

# Ignore tiny center error inside this deadband.
# Raise to reduce jitter; lower for tighter tracking.
CENTER_DEADBAND = 0.03

# If horizontal error is larger than this, linear speed is reduced.
MISALIGN_ERR_FOR_SLOWDOWN = 0.16

# Linear speed multiplier when target is significantly off-center.
# Lower value = stronger slowdown while turning.
MISALIGN_LINEAR_SCALE = 0.65


# -----------------------------------------------------------------------------
# Lost-Target Search Behavior (phased sweep + forward arcs)
# -----------------------------------------------------------------------------
# Frames with no valid target before active searching starts.
SEARCH_START_FRAMES = 7

# Base yaw rate used for ball search phases.
BALL_SEARCH_TURN_RATE = 0.22

# Base yaw rate used for bucket search phases.
BUCKET_SEARCH_TURN_RATE = 0.22

# Forward component during search arcs for balls.
# Negative value drives forward in your current sign convention.
BALL_SEARCH_FORWARD_LIN = -0.10

# Forward component during search arcs for buckets.
# Typically a little more aggressive because bucket detection is harder at range.
BUCKET_SEARCH_FORWARD_LIN = -0.10

# Duration (frames) of each search phase before switching pattern.
# Lower = quicker scan changes, higher = smoother longer sweeps.
SEARCH_PHASE_FRAMES = 24

# Turn-rate multiplier used for the "wide opposite sweep" phase.
SEARCH_WIDE_TURN_MULT = 1.45

# Turn-rate multiplier used for forward-arc phases.
SEARCH_ARC_TURN_MULT = 0.55


# -----------------------------------------------------------------------------
# Depth Robustness
# -----------------------------------------------------------------------------
# Radius of square kernel sampled around detection center for depth median.
# 2 means a 5x5 sample window.
# Raise if depth is noisy; lower if objects are tiny and nearby clutter leaks in.
DEPTH_KERNEL_RADIUS = 2


# -----------------------------------------------------------------------------
# Odometry / Camera Transform (staged merge support)
# -----------------------------------------------------------------------------
# Enable host-side odom integration and telemetry parsing from Pico feedback.
ODOM_ENABLE = True

# Camera translation relative to robot base center (validated from gb_follower tests).
ODOM_CAM_OFFSET_X_M = -0.64
ODOM_CAM_OFFSET_Y_M = 0.0
ODOM_CAM_OFFSET_Z_M = 0.2

# Odometry goal controller gains and limits.
ODOM_KP_V = 0.5
ODOM_KP_W = 0.5
ODOM_MAX_V = 0.30
ODOM_MAX_W = 0.60
ODOM_GOAL_TOLERANCE_M = 0.05

# In current robot convention, negative linear command drives forward.
ODOM_FORWARD_IS_NEGATIVE = False

# If True, enable odom-assisted velocity output in ball detect mode only.
# If False, state_machine keeps using vision steering only.
ODOM_ASSIST_BALL_ENABLE = True

# Minimum XY delta before replacing current odom goal from a new ball observation.
ODOM_GOAL_UPDATE_MIN_DELTA_M = 0.02

# Tiny-command deadband for accepting odom controller outputs.
ODOM_CMD_EPSILON = 0.01

# If True, odom-goal updates are rejected when detection color does not match
# current mission objective color (unless color fallback mode is active).
ODOM_GOAL_RESPECT_COLOR_FILTER = True

# Kernel radius used for odom-goal depth sampling around detection center.
# 0 = single pixel, 1 = 3x3 median, 2 = 5x5 median.
ODOM_GOAL_DEPTH_KERNEL_RADIUS = 1

# If True, keep the current odom goal when new candidates move only slightly.
# This reduces target jitter when multiple similar detections are clustered.
ODOM_GOAL_HYSTERESIS_ENABLE = True

# Radius (meters) inside which new goal candidates are ignored while a goal is active.
ODOM_GOAL_HYSTERESIS_M = 0.12

# If True, clear odom goal when no valid ball target is detected.
ODOM_CLEAR_GOAL_WHEN_BALL_LOST = True

# If True, enable odom-assisted velocity output in bucket detect mode.
ODOM_ASSIST_BUCKET_ENABLE = True

# If True, clear odom goal when no valid bucket target is detected.
ODOM_CLEAR_GOAL_WHEN_BUCKET_LOST = True

# If True, snapshot and reset odom debug counters after each completed lap/drop.
ODOM_RESET_COUNTERS_EACH_LAP = True
