import cv2
import numpy as np
from config import RISK_AREA_CRITICAL, RISK_AREA_HIGH, RISK_AREA_MEDIUM

class RiskEngine:
    def __init__(self):
        pass
        
    def get_driving_corridor(self, frame_shape):
        """Returns a polygon defining the driving corridor."""
        h, w = frame_shape[:2]
        # Trapezoid shape
        pt1 = (int(w * 0.40), int(h * 0.60)) # Top left
        pt2 = (int(w * 0.60), int(h * 0.60)) # Top right
        pt3 = (int(w * 0.95), h)             # Bottom right
        pt4 = (int(w * 0.05), h)             # Bottom left
        return np.array([pt1, pt2, pt3, pt4], np.int32)
        
    def evaluate_risk(self, detections, frame_shape):
        h, w = frame_shape[:2]
        frame_area = h * w
        
        highest_risk = "SAFE"
        risk_score = 0
        corridor_poly = self.get_driving_corridor(frame_shape)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            
            box_area = (x2 - x1) * (y2 - y1)
            relative_area = box_area / frame_area
            
            # Check if bottom-center of box is inside corridor
            cx = float((x1 + x2) / 2.0)
            cy = float(y2)
            in_corridor = cv2.pointPolygonTest(corridor_poly, (cx, cy), False) >= 0
            
            det_risk = "SAFE"
            
            # Very simplistic logic for PoC
            if in_corridor:
                if relative_area > RISK_AREA_CRITICAL:
                    det_risk = "CRITICAL"
                    risk_score = max(risk_score, 90)
                elif relative_area > RISK_AREA_HIGH:
                    det_risk = "HIGH"
                    risk_score = max(risk_score, 70)
                elif relative_area > RISK_AREA_MEDIUM:
                    det_risk = "MEDIUM"
                    risk_score = max(risk_score, 40)
                else:
                    det_risk = "LOW"
                    risk_score = max(risk_score, 20)
            else:
                # Outside corridor but large could still be a risk (e.g., crossing pedestrian)
                if relative_area > RISK_AREA_HIGH and cls_name in ['Person', 'Bicycle']:
                    det_risk = "MEDIUM"
                    risk_score = max(risk_score, 50)
                    
            det['risk_level'] = det_risk
            
            if det_risk == "CRITICAL":
                highest_risk = "CRITICAL"
            elif det_risk == "HIGH" and highest_risk != "CRITICAL":
                highest_risk = "WARNING"
            elif det_risk == "MEDIUM" and highest_risk not in ["CRITICAL", "WARNING"]:
                highest_risk = "CAUTION"
                
        return highest_risk, risk_score
