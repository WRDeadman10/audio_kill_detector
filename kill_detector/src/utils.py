import os
import json
from typing import List, Dict

SAMPLE_RATE = 22050
WINDOW_SECONDS = 0.5
DEFAULT_CONFIDENCE = 0.7
MIN_GAP_SECONDS = 0.3

def setup_logger(name: str) -> logging.Logger:
    "Setup logger"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def ensure_directory(path: str) -> None:
    "Ensure directory exists"
    if not os.path.exists(path):
        os.makedirs(path)

def write_json(data: Dict, path: str) -> None:
    "Write JSON data safely"
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def find_mp4_files(directory: str) -> List[str]:
    "Find all mp4 files in directory"
    return [f for f in os.listdir(directory) if f.endswith('.mp4')]

def filter_timestamps(timestamps: List[float], min_gap_seconds: float = MIN_GAP_SECONDS) -> List[float]:
    "Filter timestamps with minimum gap"
    filtered = []
    last_time = None
    for time in timestamps:
        if last_time is None or time - last_time >= min_gap_seconds:
            filtered.append(time)
            last_time = time
    return filtered
