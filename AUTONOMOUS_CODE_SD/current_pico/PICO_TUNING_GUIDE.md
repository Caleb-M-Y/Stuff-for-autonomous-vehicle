# current_pico Arm Tuning

Edit `arm_tuning.py` to tune arm travel.

## First knobs to try

- `SHOULDER_CMD_SCALE`
  Increases/decreases shoulder motion per incoming command.
  If arm does not drop low enough, increase this first.

- `SHOULDER_LOWER_SCALE`
  Extra gain only when lowering (`shoulder_cmd > 0`).
  Raise this if lowering is still too shallow.

- `SHOULDER_A_MAX`, `SHOULDER_B_MIN`
  Mechanical lower-end limits. Increase/decrease carefully to avoid binding.

- `SHOULDER_A_NEUTRAL`, `SHOULDER_B_NEUTRAL`
  Home pose.

## Safe tuning sequence

1. Increase `SHOULDER_CMD_SCALE` by small steps (for example `+0.2`).
2. Test one pickup cycle.
3. If still too high, increase `SHOULDER_LOWER_SCALE` by `+0.05`.
4. Only then adjust hard limits (`*_MAX`, `*_MIN`) in small increments.

## Runtime behavior

- `main.py` now has a command timeout deadman (`CMD_TIMEOUT_US`) so wheels/arm stop if host messages stall.
- Command parsing is non-blocking and robust to malformed lines.
