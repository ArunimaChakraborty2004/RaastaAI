import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, YOLO_MODEL_NAME, TARGET_CLASSES, DEFAULT_CONFIDENCE_THRESHOLD, CLASS_NAMES

class ObjectDetector:
    def __init__(self):
        # Initialize YOLOv8 nano model. 
        # Ultralytics will download it automatically if not present in the path.
        self.model = YOLO(YOLO_MODEL_NAME)
        # We can also save it to our models dir later if we want, but default is fine.
        
    def detect(self, frame, conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        # Run inference
        results = self.model(frame, conf=conf_threshold, classes=TARGET_CLASSES, verbose=False)
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Filter for target classes just in case
                if cls_id in TARGET_CLASSES:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_id': cls_id,
                        'class_name': CLASS_NAMES.get(cls_id, 'Unknown')
                    })
                    
        return detections
