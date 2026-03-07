# SD Data and Training Pipeline

This folder is a standalone data + training workflow for your `AUTONOMOUS_CODE_SD` competition stack.

It is designed around your current runtime assumptions:
- Active class indices must stay stable once deployed.
- Runtime label matching in `AUTONOMOUS_CODE_SD/state_machine.py` depends on `ball` and `bucket` class names.
- Hailo labels JSON should be written to `AUTONOMOUS_CODE_SD/models/ball_bucket.json`.

## Folder Contents

- `config.example.yaml`: Main config template.
- `requirements.txt`: Python packages for this pipeline.
- `scripts/extract_frames.py`: Pull images from videos.
- `scripts/roboflow_download.py`: Download Roboflow versioned dataset.
- `scripts/convert_coco_to_yolo.py`: Convert Roboflow COCO split exports into YOLO txt labels.
- `scripts/auto_annotate_yolo.py`: Pseudo-label new images using a teacher YOLO model.
- `scripts/build_review_set.py`: Build a sampled visual QA set from YOLO labels.
- `scripts/train_yolo.py`: Train a new YOLO model.
- `scripts/evaluate_yolo.py`: Validate trained weights.
- `scripts/export_for_hailo.py`: Export ONNX + generate Hailo labels JSON.
- `scripts/package_model_release.py`: Build a versioned robot-ready release bundle.
- `model_info.template.txt`: Template for release metadata.

## Quick Start

1. Create your working config:

```powershell
copy sd_data_and_training\config.example.yaml sd_data_and_training\config.yaml
```

2. Install pipeline dependencies in your existing `.venv`:

```powershell
uv pip install -r sd_data_and_training\requirements.txt
```

If your environment has `pip` available, `python -m pip install ...` also works.

3. Fill in `sd_data_and_training/config.yaml` paths:
- `paths.unlabeled_images_dir`
- `paths.train_images_dir`
- `paths.val_images_dir`
- `paths.train_labels_dir`
- `paths.val_labels_dir`
- `roboflow.*` (if using Roboflow API pull)

4. If your Roboflow export is COCO (contains `_annotations.coco.json`), convert it once before training:

```powershell
python sd_data_and_training\scripts\convert_coco_to_yolo.py ^
  --config sd_data_and_training\config.yaml ^
  --dataset-root "C:\path\to\RoboFlowDatasets" ^
  --clear-existing
```

## Typical Workflow (Tonight)

### A) Optional: Pull latest annotated set from Roboflow

```powershell
$env:ROBOFLOW_API_KEY="<your_key_here>"
python sd_data_and_training\scripts\roboflow_download.py --config sd_data_and_training\config.yaml
```

### B) Extract new frames from field videos

```powershell
python sd_data_and_training\scripts\extract_frames.py ^
  --video-dir "C:\path\to\new_videos" ^
  --out-dir "C:\path\to\new_images" ^
  --every-n-frames 8
```

### C) Auto-label newly collected images

```powershell
python sd_data_and_training\scripts\auto_annotate_yolo.py ^
  --config sd_data_and_training\config.yaml ^
  --images-dir "C:\path\to\new_images" ^
  --labels-out "C:\path\to\new_labels"
```

Recommended: upload images + labels to Roboflow and do a quick human correction pass before training.

If your raw images come from multiple subfolders and repeated filenames (for example `images_16/image_000001.jpg`, `images_17/image_000001.jpg`), keep directory structure when copying. The auto-label script preserves subfolders and writes matching nested labels.

### C2) Build a fast review subset before full retrain

```powershell
python sd_data_and_training\scripts\build_review_set.py ^
  --config sd_data_and_training\config.yaml ^
  --images-dir "C:\path\to\new_images" ^
  --labels-dir "C:\path\to\new_labels" ^
  --out-dir "C:\path\to\review_subset" ^
  --sample-n 500 ^
  --empty-sample-n 180
```

This creates:
- `review_subset/non_empty/` with drawn boxes
- `review_subset/empty/` with samples that received no boxes
- `review_subset/review_manifest.csv` for traceability

### D) Train a new model

```powershell
python sd_data_and_training\scripts\train_yolo.py --config sd_data_and_training\config.yaml
```

Default training settings in `config.yaml` are tuned for an 8GB RTX 4060 (`yolov8m`, `imgsz=640`, `batch=8`, `workers=4`).

If you hit CUDA errors such as `CUBLAS_STATUS_EXECUTION_FAILED` or memory allocation failures, lower `training.batch` first (for example `6` or `4`), then lower `training.workers`.

### E) Evaluate model quality

```powershell
python sd_data_and_training\scripts\evaluate_yolo.py ^
  --config sd_data_and_training\config.yaml ^
  --weights "C:\path\to\best.pt"
```

### F) Export for deployment and update runtime labels JSON

```powershell
python sd_data_and_training\scripts\export_for_hailo.py ^
  --config sd_data_and_training\config.yaml ^
  --weights "C:\path\to\best.pt"
```

For Hailo conversion compatibility, this pipeline defaults to `opset=11` and `simplify=false`.
You can override if needed:

```powershell
python sd_data_and_training\scripts\export_for_hailo.py ^
  --config sd_data_and_training\config.yaml ^
  --weights "C:\path\to\best.pt" ^
  --opset 11 ^
  --no-simplify
```

This writes:
- ONNX model into `paths.export_out_dir`
- Updated labels JSON into `paths.hailo_labels_json_out`

### G) Package a robot-ready model bundle

```powershell
python sd_data_and_training\scripts\package_model_release.py ^
  --config sd_data_and_training\config.yaml
```

This creates a versioned folder (and zip) under `sd_data_and_training/artifacts/model_releases/` containing:
- ONNX model
- labels JSON
- `model_info.txt` metadata

Use this bundle as your deploy handoff to the robot.

## Model Versioning (Simple)

Treat each exported model as a versioned release, not just a file copy.

- Canonical copy: the one release bundle you trust for a given model version.
- Working copies: temporary copies on your PC, laptop, USB, or robot.

Recommended release name format:
- `ballbucket-YYYYMMDD-HHMMSS-<gitsha>`
- Example: `ballbucket-20260305-145200-63027f1`

Recommended delivery workflow:
1. Train + evaluate.
2. Export (`best.onnx` + `ball_bucket.json`).
3. Package release bundle with `package_model_release.py`.
4. Upload the zip as a GitHub Release asset (canonical copy).
5. Copy the same zip to robot (direct download or USB), then extract.

## Important Class Order Rule

Do not change class order unless you intentionally retrain and redeploy everything together.

Current canonical order:
1. `blue_ball`
2. `blue_bucket`
3. `green_ball`
4. `green_bucket`
5. `red_ball`
6. `red_bucket`
7. `yellow_ball`
8. `yellow_bucket`

The currently downloaded Roboflow export includes a `sport-ball` class entry with zero annotations in train/valid/test. It is intentionally excluded from this canonical order.

## Notes for Your Runtime

- Your runtime color/target matching in `AUTONOMOUS_CODE_SD/state_machine.py` can parse labels with spaces or underscores.
- `export_for_hailo.py` writes display labels with spaces to match your current `ball_bucket.json` convention.
- If you change class names, update `classes.train_names` and `classes.hailo_display_names` together.
- If telemetry ever contains unlabeled detections (for example `;:0.62`), treat it as a label-map mismatch signal and re-export/copy a fresh `ball_bucket.json` from this pipeline before field runs.

## Suggested Next Upgrade

After this works end-to-end, add an active-learning loop:
- Auto-label with high confidence.
- Route low-confidence images to a `review_queue` folder.
- Human-correct only the queue.
- Retrain weekly with mixed old + new data.
