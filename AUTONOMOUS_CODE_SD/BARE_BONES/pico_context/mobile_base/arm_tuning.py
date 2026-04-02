"""Bare-bones arm tuning for Pico debug.

These values are tuned to keep motion controlled but still strong enough to
recover upward against arm load. The previous raise tuning was too weak in
practice, so the shoulder command shaping below gives raise a little more
authority than lower.
"""

SHOULDER_A_NEUTRAL = 1_300_000
SHOULDER_B_NEUTRAL = 1_600_000
CLAW_NEUTRAL = 1_800_000

# Keep travel conservative while we validate pickup/drop behavior.
SHOULDER_A_MIN = 1_050_000
SHOULDER_A_MAX = 1_700_000
SHOULDER_B_MIN = 1_200_000
SHOULDER_B_MAX = 1_900_000
CLAW_MIN = 1_550_000
CLAW_MAX = 2_200_000

# Shoulder motion needed higher authority to reliably raise back to neutral.
SHOULDER_CMD_SCALE = 0.50
CLAW_CMD_SCALE = 0.35
SHOULDER_LOWER_SCALE = 0.50
SHOULDER_RAISE_SCALE = 0.20
MAX_ABS_HOST_CMD = 8_000

# Hard slew-rate limits per control tick to avoid slamming.
SHOULDER_STEP_LIMIT_NS = 12_000
CLAW_STEP_LIMIT_NS = 10_000

# Smooth return-to-neutral shaping (used when arm_state=10).
NEUTRAL_SHOULDER_STEP_NS = 8_000
NEUTRAL_CLAW_STEP_NS = 8_000
