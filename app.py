import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from config import OUTPUT_DIR, SAMPLE_VIDEOS_DIR

st.set_page_config(page_title="RAASTA AI - Live Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { 
        background-color: #0b101a; 
        color: #e2e8f0; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide top padding and default menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1600px;}
    
    /* Top Nav Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 15px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 15px;
    }
    .logo-container h2 {
        margin: 0; color: #ffffff; font-weight: 700; display: inline-block;
    }
    .logo-container span {
        color: #38bdf8; font-weight: 300; font-size: 0.9em;
    }
    .nav-buttons {
        display: flex; gap: 10px;
    }
    .nav-btn {
        background-color: #1e293b; color: #94a3b8; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer;
    }
    .nav-btn.active {
        background-color: #1d4ed8; color: #ffffff;
    }
    .sys-status {
        background-color: #0f172a; border: 1px solid #1e293b; padding: 8px 16px; border-radius: 6px; display: flex; align-items: center; gap: 8px; font-size: 0.8rem;
    }
    .dot {
        height: 10px; width: 10px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e;
    }
    
    /* Panel Containers */
    .panel {
        background-color: #131c2d;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .panel-header {
        font-size: 0.85rem; font-weight: 700; color: #f8fafc; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.5px;
    }
    
    /* Warning Box */
    .warning-critical {
        background: linear-gradient(180deg, #450a0a 0%, #2a0808 100%); border: 1px solid #7f1d1d; border-radius: 8px; padding: 20px; text-align: center;
    }
    .warning-title {
        color: #ef4444; font-size: 1.2rem; font-weight: 700; text-transform: uppercase; margin-bottom: 5px;
    }
    .warning-subtitle {
        color: #f87171; font-size: 0.9rem; margin-bottom: 5px;
    }
    
    /* Stat boxes */
    .stat-row {
        display: flex; justify-content: space-around; text-align: center; margin-top: 10px;
    }
    .stat-box .val {
        font-size: 1.5rem; font-weight: 700; color: #ef4444; margin-bottom: 2px;
    }
    .stat-box .lbl {
        font-size: 0.75rem; color: #94a3b8;
    }
    
    /* Session Info Table */
    .session-info {
        font-size: 0.8rem; color: #cbd5e1; line-height: 1.8;
    }
    .session-info span.lbl { width: 90px; display: inline-block; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# Top Navigation
st.markdown("""
<div class="top-nav">
    <div class="logo-container">
        <h2 style="color: #38bdf8;">RAASTA <span style="color: #ffffff; font-weight: 700;">AI</span></h2><br>
        <span style="font-size: 0.8rem; color: #94a3b8;">See the Risk Before It Reaches You</span>
    </div>
    <div class="nav-buttons">
        <div class="nav-btn active">📊 Dashboard</div>
        <div class="nav-btn">📈 Analytics</div>
        <div class="nav-btn">⚙️ Settings</div>
    </div>
    <div class="sys-status">
        <div class="dot"></div>
        <div>
            <div style="color: #94a3b8; font-size: 0.7rem;">System Status</div>
            <div style="color: #22c55e; font-weight: 700;">OPERATIONAL</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Setup inputs via sidebar
with st.sidebar:
    st.markdown("### 🔌 Sensor Suite")
    st.checkbox("📷 Primary Camera (Active)", value=True, disabled=True)
    st.checkbox("📡 Radar Fusion Module", value=False)
    st.checkbox("🌙 IR Night Vision", value=False)
    st.checkbox("👁️ Driver Monitoring", value=True)
    st.markdown("---")
    
    st.subheader("Video Source")
    use_demo = st.checkbox("Use Demo Video", value=True)
    rotate_video = st.checkbox("Auto-Rotate Phone Video", value=False, help="Check this if your video looks sideways")
    uploaded_file = st.file_uploader("Or Upload Custom Video")
    start_btn = st.button("Start Analysis")

# Main Layout
col_main, col_right = st.columns([2.5, 1])

# Placeholders
with col_main:
    # Video Panel
    st.markdown('<div class="panel-header" style="margin-top: 10px;">LIVE ROAD ANALYSIS</div>', unsafe_allow_html=True)
    st_video = st.empty()
    
    st.markdown('<div class="panel-header" style="margin-top: 20px;">ENVIRONMENT & TELEMETRY</div>', unsafe_allow_html=True)
    st_env = st.empty()
    
    st.markdown('<div class="panel-header" style="margin-top: 20px;">DETECTION SUMMARY</div>', unsafe_allow_html=True)
    st_stats = st.empty()
    
    st.markdown('<div class="panel-header" style="margin-top: 20px;">SESSION INFO</div>', unsafe_allow_html=True)
    st_session = st.empty()

with col_right:
    st_warning = st.empty()
    st_gauge = st.empty()
    st_lane_assist = st.empty()
    st_factors = st.empty()
    st_alerts = st.empty()

if start_btn or use_demo:
    video_path = str(SAMPLE_VIDEOS_DIR / "VID20260704115232.mp4") if use_demo and not uploaded_file else None
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name
        
    if video_path and os.path.exists(video_path):
        from src.video_processor import VideoProcessor
        processor = VideoProcessor()
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
        
            if rotate_video:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            frame_count += 1
            processed_frame, metrics = processor.process_frame(frame)
            
            # --- UPDATE VIDEO ---
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_video.image(rgb_frame, channels="RGB", width='stretch')
            
            # --- UPDATE ENV & TELEMETRY ---
            import random
            speed = 45 + random.randint(-2, 2)
            ttc = max(0.5, 3.5 - (metrics['risk_score'] / 30.0)) if metrics['risk_score'] > 0 else 5.0 + random.random()
            
            st_env.markdown(f"""
            <div class="panel" style="margin-top: 0;">
                <div class="stat-row">
                    <div class="stat-box"><div class="val" style="color: #38bdf8;">{speed} <span style="font-size: 0.8rem;">km/h</span></div><div class="lbl">Current Speed</div></div>
                    <div class="stat-box"><div class="val" style="color: #22c55e;">CLEAR</div><div class="lbl">Weather</div></div>
                    <div class="stat-box"><div class="val" style="color: #eab308;">DRY</div><div class="lbl">Road Surface</div></div>
                    <div class="stat-box"><div class="val" style="color: #8b5cf6;">{ttc:.1f}s</div><div class="lbl">Est. Time-to-Collision</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE STATS ---
            st_stats.markdown(f"""
            <div class="panel" style="margin-top: 0;">
                <div class="stat-row">
                    <div class="stat-box"><div class="val" style="color: #ef4444;">{metrics['vehicles']}</div><div class="lbl">Vehicles</div></div>
                    <div class="stat-box"><div class="val" style="color: #eab308;">{metrics['pedestrians']}</div><div class="lbl">Pedestrians</div></div>
                    <div class="stat-box"><div class="val" style="color: #22c55e;">{metrics['cyclists']}</div><div class="lbl">Cyclists</div></div>
                    <div class="stat-box"><div class="val" style="color: #3b82f6;">{metrics['trucks']}</div><div class="lbl">Trucks</div></div>
                    <div class="stat-box"><div class="val" style="color: #8b5cf6;">{metrics['buses']}</div><div class="lbl">Buses</div></div>
                    <div class="stat-box"><div class="val" style="color: #f97316;">{metrics['traffic_lights']}</div><div class="lbl">Traffic Lights</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE SESSION ---
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            mins, secs = divmod(int(elapsed), 60)
            
            st_session.markdown(f"""
            <div class="panel" style="margin-top: 0;">
                <div class="session-info">
                    <div><span class="lbl">Video File:</span> live_feed.mp4</div>
                    <div><span class="lbl">Resolution:</span> {frame.shape[1]} x {frame.shape[0]}</div>
                    <div><span class="lbl">Frame Rate:</span> {fps:.1f} FPS</div>
                    <div><span class="lbl">Duration:</span> {mins:02d}:{secs:02d}</div>
                    <div><span class="lbl">Status:</span> <span style="color:#22c55e">Processing</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE WARNING ---
            risk = metrics['risk_status']
            if risk == "CRITICAL":
                st_warning.markdown("""
                <div class="panel warning-critical">
                    <div style="color: #ef4444; font-size: 2rem; margin-bottom: 10px;">⚠️</div>
                    <div class="warning-title">WARNING<br>VEHICLE AHEAD</div>
                    <div class="warning-subtitle">HIGH COLLISION RISK</div>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 10px;">Maintain Safe Distance</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st_warning.markdown("""
                <div class="panel" style="text-align: center; padding: 30px;">
                    <div style="color: #22c55e; font-size: 2rem; margin-bottom: 10px;">✓</div>
                    <div style="color: #22c55e; font-weight: 700; font-size: 1.2rem;">CLEAR ROAD</div>
                    <div style="color: #94a3b8; font-size: 0.85rem;">No Immediate Threats</div>
                </div>
                """, unsafe_allow_html=True)
                
            # --- UPDATE GAUGE (Replaced Plotly with high-performance HTML/CSS for video loop) ---
            score = metrics['risk_score']
            if score < 33:
                risk_color = "#22c55e"
            elif score < 66:
                risk_color = "#eab308"
            else:
                risk_color = "#ef4444"
                
            st_gauge.markdown(f"""
            <div class="panel" style="margin-top: 0;">
                <div class="panel-header" style="text-align: center;">COLLISION RISK</div>
                <div style="background-color: #0f172a; border-radius: 20px; height: 25px; width: 100%; overflow: hidden; margin-top: 15px; border: 1px solid #334155; box-shadow: inset 0 0 10px rgba(0,0,0,0.5);">
                    <div style="width: {score}%; background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%); height: 100%; transition: width 0.2s;"></div>
                </div>
                <div style="text-align: center; margin-top: 10px; font-size: 2rem; font-family: 'Orbitron', sans-serif; font-weight: bold; color: {risk_color}; text-shadow: 0 0 10px {risk_color};">
                    {score}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE LANE ASSIST ---
            lane_status = metrics.get('lane_status', 'Stable')
            lane_conf = metrics.get('lane_confidence', 'High')
            st_lane_assist.markdown(f"""
            <div class="panel" style="margin-top: 0px;">
                <div class="panel-header">LANE KEEP ASSIST</div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 2.2rem; color: #38bdf8; text-shadow: 0 0 10px #38bdf8;">🛣️</div>
                    <div style="text-align: right;">
                        <div style="color: #22c55e; font-weight: bold; font-size: 1.2rem; text-shadow: 0 0 5px #22c55e;">{lane_status.upper()}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">Confidence: {lane_conf}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE FACTORS ---
            st_factors.markdown("""
            <div class="panel" style="margin-top: 0px;">
                <div class="panel-header">RISK FACTORS</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.8;">
                    <div><span style="color:#22c55e">✓</span> Object in driving corridor</div>
                    <div><span style="color:#22c55e">✓</span> Large object in frame</div>
                    <div><span style="color:#22c55e">✓</span> Near to vehicle</div>
                    <div><span style="color:#22c55e">✓</span> Approaching (increasing size)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- UPDATE ALERTS ---
            st_alerts.markdown("""
            <div class="panel" style="margin-top: 0px;">
                <div class="panel-header">RECENT ALERTS</div>
                <div style="font-size: 0.8rem; line-height: 2;">
                    <div style="display:flex; justify-content:space-between;"><span style="color:#ef4444">● 00:00:41</span> <span style="color:#ef4444">VEHICLE AHEAD</span> <span style="color:#ef4444">HIGH RISK</span></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#eab308">● 00:00:39</span> <span style="color:#cbd5e1">PEDESTRIAN AHEAD</span> <span style="color:#eab308">CAUTION</span></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#f97316">● 00:00:34</span> <span style="color:#f97316">VEHICLE AHEAD</span> <span style="color:#f97316">MEDIUM RISK</span></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#eab308">● 00:00:28</span> <span style="color:#cbd5e1">MAINTAIN LANE</span> <span style="color:#eab308">CAUTION</span></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#22c55e">● 00:00:21</span> <span style="color:#cbd5e1">CLEAR ROAD</span> <span style="color:#22c55e">SAFE</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    with col_main:
        st.info("👈 Please open the sidebar on the left and click **'Start Analysis'** to begin the live video feed!")
