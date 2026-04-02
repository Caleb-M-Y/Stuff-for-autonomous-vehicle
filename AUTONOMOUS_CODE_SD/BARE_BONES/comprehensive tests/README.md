# Comprehensive Tests (Bare Bones)

This folder contains focused, staged tests to validate each subsystem before running the full autonomous stack.

Run from your Pi project root:

```bash
cd /home/ball-e/ball-e/python_scripts/BARE_BONES/comprehensive\ tests
```

Recommended order:

1. `test_01_camera_stream_exact.py`
2. `test_02_model_pipeline_smoke.py`
3. `test_03_serial_protocol_5field.py`
4. `test_04_wheel_motion_basic.py`
5. `test_05_arm_motion_basic.py`
6. `test_06_state_machine_dry_run.py`

Notes:

- Keep wheels off the ground for wheel tests.
- Keep arm clear of obstacles for arm tests.
- The live command protocol is 5-field host command and 2-field Pico feedback.
