import time

class WarningEngine:
    def __init__(self):
        self.current_warning = "SAFE"
        self.warning_text = ""
        self.last_warning_time = 0
        self.cooldown_seconds = 2.0
        
    def process_state(self, highest_risk, detections):
        current_time = time.time()
        
        # Determine base warning text
        new_warning = ""
        if highest_risk == "CRITICAL":
            new_warning = "COLLISION RISK!"
            for det in detections:
                if det.get('risk_level') == "CRITICAL":
                    if det['class_name'] == 'Person':
                        new_warning = "CRITICAL: PEDESTRIAN AHEAD"
                    elif det['class_name'] == 'Bicycle':
                        new_warning = "CRITICAL: CYCLIST AHEAD"
                    else:
                        new_warning = "CRITICAL: COLLISION IMMINENT"
                    break
        elif highest_risk == "WARNING":
            new_warning = "WARNING: MAINTAIN SAFE DISTANCE"
        elif highest_risk == "CAUTION":
            new_warning = "CAUTION: HAZARD DETECTED"
        else:
            new_warning = "SYSTEM ACTIVE"
            
        # Update logic with cooldown
        if highest_risk in ["CRITICAL", "WARNING"]:
            self.current_warning = highest_risk
            self.warning_text = new_warning
            self.last_warning_time = current_time
        else:
            if current_time - self.last_warning_time > self.cooldown_seconds:
                self.current_warning = highest_risk
                self.warning_text = new_warning
                
        return self.current_warning, self.warning_text
