# SD Data and Training Pipeline

This folder is a standalone data + training workflow for your `AUTONOMOUS_CODE_SD` competition stack.

It is designed around your current runtime assumptions:
- 8 class indices must stay stable.
- Runtime label matching in `AUTONOMOUS_CODE_SD/state_machine.py` depends on `ball` and `bucket` class names.
- Hailo labels JSON should be written to `AUTONOMOUS_CODE_SD/models/ball_bucket.json`.

## Folder Contents

- `config.example.yaml`: Main config template.
- `requirements.txt`: Python packages for this pipeline.
- `scripts/extract_frames.py`: Pull images from videos.
- `scripts/roboflow_download.py`: Download Roboflow versioned dataset.
- `scripts/auto_annotate_yolo.py`: Pseudo-label new images using a teacher YOLO model.
- `scripts/train_yolo.py`: Train a new YOLO model.
- `scripts/evaluate_yolo.py`: Validate trained weights.
- `scripts/export_for_hailo.py`: Export ONNX + generate Hailo labels JSON.

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

### D) Train a new model

```powershell
python sd_data_and_training\scripts\train_yolo.py --config sd_data_and_training\config.yaml
```

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

This writes:
- ONNX model into `paths.export_out_dir`
- Updated labels JSON into `paths.hailo_labels_json_out`

## Important Class Order Rule

Do not change class order unless you intentionally retrain and redeploy everything together.

Current canonical order:
1. `blue_bucket`
2. `blue_ball`
3. `yellow_bucket`
4. `green_ball`
5. `green_bucket`
6. `red_ball`
7. `red_bucket`
8. `yellow_ball`

## Notes for Your Runtime

- Your runtime color/target matching in `AUTONOMOUS_CODE_SD/state_machine.py` can parse labels with spaces or underscores.
- `export_for_hailo.py` writes display labels with spaces to match your current `ball_bucket.json` convention.
- If you change class names, update `classes.train_names` and `classes.hailo_display_names` together.

## Suggested Next Upgrade

After this works end-to-end, add an active-learning loop:
- Auto-label with high confidence.
- Route low-confidence images to a `review_queue` folder.
- Human-correct only the queue.
- Retrain weekly with mixed old + new data.
