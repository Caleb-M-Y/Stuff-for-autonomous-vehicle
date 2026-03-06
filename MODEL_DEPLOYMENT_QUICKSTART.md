# Model Deployment Quickstart

This is the simple workflow for getting an improved model onto your robot.

## What "Canonical Copy" Means

- Canonical copy = the one official model package for a version.
- Working copies = temporary copies on home PC, laptop, USB, or robot.

Use one versioned package per model release so you always know which model is on the robot.

## Release Naming

Use this format:
- `ballbucket-YYYYMMDD-HHMMSS-<gitsha>`

Example:
- `ballbucket-20260306-012440-63027f1`

## Build A Robot-Ready Bundle

From repo root:

```powershell
python sd_data_and_training\scripts\package_model_release.py --config sd_data_and_training\config.yaml
```

This creates:
- `sd_data_and_training/artifacts/model_releases/<release_id>/best.onnx`
- `sd_data_and_training/artifacts/model_releases/<release_id>/ball_bucket.json`
- `sd_data_and_training/artifacts/model_releases/<release_id>/model_info.txt`
- `sd_data_and_training/artifacts/model_releases/<release_id>.zip`

## Important Runtime Note

Your runtime scripts in `AUTONOMOUS_CODE_SD` use Hailo `.hef` files (plus labels JSON), not raw ONNX directly.

Examples in code:
- `AUTONOMOUS_CODE_SD/camera_test2.py` default `--hef-path`
- `AUTONOMOUS_CODE_SD/course_autonomous_depth.py` default `--hef`

So deployment is usually:
1. Train and export ONNX.
2. Compile ONNX to Hailo `.hef` using your Hailo toolchain.
3. Place `.hef` and `ball_bucket.json` on robot (often under `AUTONOMOUS_CODE_SD/models/`).

## Should You Use GitHub Or USB?

Best practice is both:

1. GitHub Release asset: canonical backup and version history.
2. USB transfer: fastest way to move files onto robot when convenient.

If available, upload `<release_id>.zip` to a GitHub Release, then either:
- download it on robot, or
- copy it by USB and extract.

## Minimum Files Needed On Robot

- Compiled model: `<release_id>.hef`
- Labels file: `ball_bucket.json`
- Optional but useful: `model_info.txt` for traceability
