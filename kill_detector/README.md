# Audio Kill Detector

## Overview
Audio-based kill event detector for Valorant clips.

## Architecture
1. `src/audio_extractor.py` converts video to mono WAV (22050Hz).
2. `src/onset_detector.py` finds candidate events with `librosa.onset.onset_detect`.
3. `src/feature_extractor.py` builds fixed-size feature vectors.
4. `src/predict.py` runs model scoring + confidence/gap filtering.
5. `src/train_model.py` trains `RandomForestClassifier` from labeled samples.
6. `main.py` batch CLI writes one JSON per input video.

## Requirements
- Python 3.10+
- FFmpeg installed and available on PATH
- Install Python deps:

```bash
pip install -r requirements.txt
```

## Data Layout
- `data/samples/kill`
- `data/samples/non_kill`

## Train
```bash
python src/train_model.py
```

## Inference
```bash
python main.py --input ./data/raw --output ./output
```

## Output JSON
```json
{
  "video": "clip1.mp4",
  "kills": [12.43, 28.91, 45.02]
}
```

## Troubleshooting
- FFmpeg not found: install FFmpeg and add to PATH.
- No detections: lower `--confidence` or improve sample quality.
- Too many detections: increase `--confidence` or `--min-gap`.

## Performance Notes
- Process files in batch mode from one input folder.
- Keep audio extraction cached where possible.
- Use concise sample sets for quick training iterations.