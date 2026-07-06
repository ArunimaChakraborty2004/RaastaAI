from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Model
YOLO_MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# Classes relevant for ADAS
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
CLASS_NAMES = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck',
    9: 'Traffic Light',
    11: 'Stop Sign'
}

# UI Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

# Risk thresholds (area relative to frame)
RISK_AREA_CRITICAL = 0.15
RISK_AREA_HIGH = 0.05
RISK_AREA_MEDIUM = 0.01

# Colors (BGR for OpenCV) based on the target design
COLORS = {
    'SAFE': (0, 255, 0),
    'CAUTION': (0, 255, 255),
    'WARNING': (0, 165, 255),
    'CRITICAL': (0, 0, 255),
    'CORRIDOR': (144, 238, 144), # Light green
    'CORRIDOR_FILL': (0, 100, 0) # Dark green for transparency
}

# Class Specific BBox Colors (BGR)
CLASS_COLORS = {
    'Car': (0, 0, 255),         # Red
    'Motorcycle': (203, 10, 255), # Magenta/Pink
    'Bus': (255, 100, 0),       # Blue (OpenCV BGR so 255, 100, 0 is light blue)
    'Person': (0, 215, 255),    # Yellow
    'Truck': (255, 0, 0),       # Deep Blue
    'Bicycle': (0, 165, 255),   # Orange
    'Traffic Light': (0, 255, 255), # Yellow
    'Stop Sign': (0, 0, 255)
}
