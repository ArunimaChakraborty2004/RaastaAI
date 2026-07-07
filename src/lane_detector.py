import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        pass
        
    def detect_lanes(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. ROI
        polygon = np.array([
            [(int(w * 0.1), h), (int(w * 0.45), int(h * 0.6)), (int(w * 0.55), int(h * 0.6)), (int(w * 0.9), h)]
        ])
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Hough Lines
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=100)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                coords = np.asarray(line).reshape(-1)
                if coords.size < 4:
                    continue
                x1, y1, x2, y2 = coords[:4].astype(int)
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                if slope < -0.3: # Left lane
                    left_lines.append((slope, intercept))
                elif slope > 0.3: # Right lane
                    right_lines.append((slope, intercept))
                    
        # Average lines
        left_lane = np.average(left_lines, axis=0) if len(left_lines) > 0 else None
        right_lane = np.average(right_lines, axis=0) if len(right_lines) > 0 else None
        
        line_image = np.zeros_like(frame)
        
        def make_points(y1, y2, line):
            slope, intercept = line
            if abs(slope) < 1e-6:
                return None
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            return ((x1, y1), (x2, y2))
            
        y1 = h
        y2 = int(h * 0.6)
        
        lane_status = "Stable"
        lane_confidence = "High"
        
        if left_lane is not None:
            points = make_points(y1, y2, left_lane)
            if points:
                p1, p2 = points
                cv2.line(line_image, p1, p2, (255, 0, 0), 5)
        else:
            lane_confidence = "Low"
            
        if right_lane is not None:
            points = make_points(y1, y2, right_lane)
            if points:
                p1, p2 = points
                cv2.line(line_image, p1, p2, (255, 0, 0), 5)
        else:
            lane_confidence = "Low"
            
        if left_lane is None and right_lane is None:
            lane_status = "Not Detected"
            lane_confidence = "None"
            
        # Blend
        result = cv2.addWeighted(frame, 1, line_image, 1, 0)
        
        return result, {"lane_status": lane_status, "confidence": lane_confidence}
