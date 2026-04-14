You are a senior ML + audio signal processing engineer.

Build a complete Python project that detects "kill events" from Valorant gameplay videos using audio classification.

The system must be modular, production-ready, and optimized for accuracy + speed.

----------------------------------
PROJECT GOAL
----------------------------------
Input:
- Folder of video files (.mp4)

Output:
- JSON file per video with detected kill timestamps (in seconds)
- Example:
  {
    "video": "clip1.mp4",
    "kills": [12.43, 28.91, 45.02]
  }

----------------------------------
CORE APPROACH
----------------------------------
Use a hybrid pipeline:

1. Extract audio from video (FFmpeg)
2. Perform onset detection to find candidate sound events
3. For each candidate:
   - Extract audio window (~0.5 sec)
   - Convert to features (MFCC + spectral features)
4. Run a trained ML classifier (RandomForest)
5. Filter predictions using:
   - confidence threshold
   - minimum time gap between kills (>= 300ms)

----------------------------------
PROJECT STRUCTURE
----------------------------------

/kill_detector
│
├── data/
│   ├── raw/              # raw videos
│   ├── processed/        # extracted audio
│   ├── samples/
│   │   ├── kill/
│   │   └── non_kill/
│
├── models/
│   └── kill_model.pkl
│
├── src/
│   ├── audio_extractor.py
│   ├── onset_detector.py
│   ├── feature_extractor.py
│   ├── train_model.py
│   ├── predict.py
│   ├── utils.py
│
├── main.py
├── requirements.txt
└── README.md

----------------------------------
IMPLEMENTATION DETAILS
----------------------------------

1. audio_extractor.py
- Use ffmpeg-python
- Extract mono WAV audio (22050 Hz)

2. onset_detector.py
- Use librosa.onset.onset_detect
- Return timestamps of candidate events

3. feature_extractor.py
Extract:
- MFCC (13 coefficients, mean + std)
- spectral centroid
- zero crossing rate
- spectral rolloff

Return fixed-length numpy vector

4. train_model.py
- Load dataset from:
    data/samples/kill
    data/samples/non_kill
- Extract features
- Train RandomForestClassifier (n_estimators=100)
- Save model to models/kill_model.pkl
- Print accuracy, precision, recall

5. predict.py
Pipeline:
- Load model
- Extract audio
- Run onset detection
- For each onset:
    - extract 0.5 sec window
    - extract features
    - predict probability
- Keep events with prob > 0.7
- Apply time-gap filtering (>= 0.3 sec)

6. main.py
- CLI interface:
    python main.py --input ./data/raw --output ./output

----------------------------------
EXTRA REQUIREMENTS
----------------------------------

- Use clean OOP design where needed
- Add logging
- Handle edge cases (short audio, empty clips)
- Ensure fast batch processing

----------------------------------
OPTIONAL (IF POSSIBLE)
----------------------------------

- Add confidence score per kill
- Add visualization (plot waveform + detected points)
- Add ability to retrain model easily

----------------------------------
IMPORTANT
----------------------------------

Do NOT use deep learning.
Use only:
- librosa
- numpy
- scikit-learn
- ffmpeg-python

----------------------------------
OUTPUT FORMAT
----------------------------------

Generate:
1. Full working code (all files)
2. requirements.txt
3. README with setup + usage instructions