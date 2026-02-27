"""
Central tuning values for autonomous behavior.
Edit only this file for quick field tuning.
Units:
- Distance: meters
- Velocity: same units used by Pico command protocol
- Angles: rad/s style command magnitudes
"""

# Detection quality
MIN_CONFIDENCE = 0.45
CONFIRM_FRAMES = 3
CLOSE_CONFIRM_FRAMES = 3

# Geometry fallback
FOCAL_PX = 3386.0
BALL_HEIGHT_M = 0.1524
BUCKET_HEIGHT_M = 0.381
MIN_VALID_DEPTH_M = 0.10

# Ball approach and pickup
BALL_FAR_M = 3.0
BALL_MID_M = 1.4
BALL_PICK_MIN_M = 0.55
BALL_PICK_MAX_M = 0.70
BALL_TOO_CLOSE_BACKUP_LIN = 0.08
BALL_SPEED_FAR = -0.28
BALL_SPEED_MID = -0.18
BALL_SPEED_NEAR = -0.10
BALL_CENTER_TOL_FOR_PICK = 0.08

# Bucket approach and drop
BUCKET_FAR_M = 4.0
BUCKET_MID_M = 2.0
BUCKET_DROP_MIN_M = 0.45
BUCKET_DROP_MAX_M = 0.80
BUCKET_TOO_CLOSE_BACKUP_LIN = 0.06
BUCKET_SPEED_FAR = -0.22
BUCKET_SPEED_MID = -0.14
BUCKET_SPEED_NEAR = -0.08
BUCKET_CENTER_TOL_FOR_DROP = 0.10

# Steering and alignment
STEER_KP = 1.6
MAX_TURN_RATE = 0.55
CENTER_DEADBAND = 0.03
MISALIGN_ERR_FOR_SLOWDOWN = 0.16
MISALIGN_LINEAR_SCALE = 0.65

# Lost-target behavior
SEARCH_START_FRAMES = 7
BALL_SEARCH_TURN_RATE = 0.22
BUCKET_SEARCH_TURN_RATE = 0.22

# Depth median sampling (square kernel around bbox center)
DEPTH_KERNEL_RADIUS = 2
