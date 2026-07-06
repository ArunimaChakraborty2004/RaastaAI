import cv2
import time
import json
import threading
from flask import Flask, Response, render_template, jsonify
from src.video_processor import VideoProcessor
from config import SAMPLE_VIDEOS_DIR

app = Flask(__name__)

# Global state for telemetry
telemetry_data = {
    "speed": 0,
    "ttc": 5.0,
    "metrics": {
        "vehicles": 0,
        "pedestrians": 0,
        "cyclists": 0,
        "trucks": 0,
        "buses": 0,
        "traffic_lights": 0,
        "risk_score": 0,
        "risk_status": "NORMAL",
        "lane_status": "Stable",
        "lane_confidence": "High"
    },
    "alerts": []
}
telemetry_lock = threading.Lock()

processor = VideoProcessor()
video_path = str(SAMPLE_VIDEOS_DIR / "VID20260704115232.mp4")

def generate_frames():
    global telemetry_data
    cap = cv2.VideoCapture(video_path)
    
    # Track alerts so we don't spam
    last_alert_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop the video for the dashboard demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Process the frame
        processed_frame, metrics = processor.process_frame(frame)
        
        # Update telemetry data safely
        with telemetry_lock:
            import random
            speed = 45 + random.randint(-2, 2)
            risk_score = metrics.get('risk_score', 0)
            ttc = max(0.5, 3.5 - (risk_score / 30.0)) if risk_score > 0 else 5.0 + random.random()
            
            telemetry_data['speed'] = speed
            telemetry_data['ttc'] = round(ttc, 1)
            telemetry_data['metrics'] = metrics
            
            # Simple alert generation
            current_time = time.time()
            if metrics.get('risk_status') == 'CRITICAL' and current_time - last_alert_time > 3:
                alert = {"time": time.strftime("%H:%M:%S"), "message": "VEHICLE AHEAD", "level": "CRITICAL"}
                telemetry_data['alerts'].insert(0, alert)
                telemetry_data['alerts'] = telemetry_data['alerts'][:5] # keep last 5
                last_alert_time = current_time
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Small sleep to simulate realistic 30 FPS if the processor is too fast
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    with telemetry_lock:
        return jsonify(telemetry_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
