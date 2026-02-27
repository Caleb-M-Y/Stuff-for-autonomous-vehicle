"""
Pico-side arm tuning.
Edit this file to tune how far/fast the arm and claw move.
All pulse values are in nanoseconds.
"""

# Neutral positions
SHOULDER_A_NEUTRAL = 1_300_000  # left shoulder servo
SHOULDER_B_NEUTRAL = 1_600_000  # right shoulder servo
CLAW_NEUTRAL = 1_800_000

# Mechanical limits
SHOULDER_A_MIN = 400_000
SHOULDER_A_MAX = 2_500_000
SHOULDER_B_MIN = 400_000
SHOULDER_B_MAX = 2_500_000
CLAW_MIN = 1_800_000
CLAW_MAX = 2_600_000

# Command shaping from host values (e.g., shoulder cmd 3000)
SHOULDER_CMD_SCALE = 2.5
CLAW_CMD_SCALE = 1.0

# Optional directional bias (lowering usually needs more travel than raising)
SHOULDER_LOWER_SCALE = 1.15   # applied when shoulder cmd > 0
SHOULDER_RAISE_SCALE = 1.00   # applied when shoulder cmd < 0

# Safety
MAX_ABS_HOST_CMD = 50_000
