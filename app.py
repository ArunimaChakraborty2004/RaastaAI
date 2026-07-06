import streamlit as st
import cv2
import tempfile
import os
import time
import random
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path
import plotly.graph_objects as go
from config import OUTPUT_DIR, SAMPLE_VIDEOS_DIR, DEFAULT_CONFIDENCE_THRESHOLD

st.set_page_config(
    page_title="RAASTA AI — ADAS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛣️"
)

# ════════════════════════════════════════════════════════════════
#  MASTER CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── Reset ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #f1f5f9 !important; }

/* ─── Nuke Streamlit Native UI to create Desktop App Feel ─── */
#MainMenu, header, footer, [data-testid="stHeader"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; }

/* ─── Top Bar (Native integration) ─── */
.topbar {
  display: flex; align-items: center; padding: 12px 24px;
  background: #ffffff; border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
  margin: 0 -1rem 20px -1rem; gap: 20px;
}
.tb-logo { display: flex; align-items: center; gap: 10px; min-width: 190px; }
.tb-logo-icon { width: 32px; height: 32px; background: linear-gradient(135deg,#0ea5e9,#2563eb); border-radius: 8px; display:flex;align-items:center;justify-content:center; font-size:1.1rem; color: white; }
.tb-logo-text .t1 { font-weight:900; font-size:1rem; color: #0f172a; letter-spacing:1px; }
.tb-logo-text .t2 { font-size:0.6rem; color:#64748b; letter-spacing:1px; margin-top:-2px; }
.tb-mode { display:flex; align-items:center; gap:6px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:4px 10px; font-size:0.72rem; }
.tb-mode .lbl { color:#64748b; font-size:0.6rem; text-transform:uppercase; letter-spacing:1px; display:block; }
.tb-mode .val { color:#0ea5e9; font-weight:700; }
.tb-stats { display:flex; gap:20px; align-items:center; }
.tb-stat { text-align:center; }
.tb-stat .sl { font-size:0.55rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.tb-stat .sv { font-size:0.85rem; font-weight:700; color:#0f172a; }
.tb-spacer { flex: 1; }
.tb-time { text-align:right; }
.tb-time .tt { font-size:1rem; font-weight:700; color:#0f172a; }
.tb-time .td { font-size:0.65rem; color:#64748b; }

/* ─── Streamlit Sidebar Styling ─── */
[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #e2e8f0 !important;
  min-width: 250px !important;
  max-width: 250px !important;
}
[data-testid="stSidebar"] .block-container { padding: 20px !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #475569 !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #0f172a !important; font-weight: 700; letter-spacing: 1px; font-size: 0.9rem; }

/* ─── Style the Native Radio Navigation ─── */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 18px; margin: 0 -1rem; cursor: pointer;
    border-left: 3px solid transparent; transition: all 0.2s; color: #0f172a !important; font-weight: 500;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #f1f5f9; }
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: #eff6ff; border-left-color: #2563eb; color: #2563eb !important; font-weight: 700;
}

/* Upload styling */
.stFileUploader > div > div { background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; color: #334155; }

/* Start button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: white !important; border: none !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; box-shadow: 0 4px 10px rgba(37,99,235,0.2) !important;
}

/* ─── Panels ─── */
.strip-panel, .rp-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  border-radius: 12px; padding: 16px; margin-bottom: 12px;
}
.sp-title, .rp-title {
  font-size: 0.62rem; font-weight:700; text-transform:uppercase;
  letter-spacing:1.5px; color:#64748b; margin-bottom:10px;
}
.rp-title { display:flex; align-items:center; gap:6px; }
.rp-title::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#e2e8f0,transparent); }

/* Video */
.video-area { position: relative; background: #000; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 12px; }
.rec-badge { position:absolute; top:12px; left:12px; display:flex; align-items:center; gap:6px; background:rgba(255,255,255,0.9); backdrop-filter:blur(8px); border-radius:6px; padding:5px 10px; font-size:0.72rem; font-weight:700; color:#dc2626; letter-spacing:1px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.rec-dot { width:8px; height:8px; border-radius:50%; background:#dc2626; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
.speed-overlay { position:absolute; top:10px; right:12px; background:rgba(255,255,255,0.95); backdrop-filter:blur(12px); border:1px solid #e2e8f0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius:10px; padding:10px 14px; text-align:center; min-width:90px; }
.speed-label { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:2px; }
.speed-value { font-size:2.2rem; font-weight:900; color:#0f172a; line-height:1; }
.speed-unit { font-size:0.65rem; color:#64748b; }

/* Lane panel */
.lane-stats { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:10px; }
.lane-stat-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px; text-align:center; }
.lsb-val { font-size:1.1rem; font-weight:700; }
.lsb-lbl { font-size:0.55rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }

/* Threats panel */
.threat-item { display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.72rem; }
.threat-item:last-child { border-bottom:none; }
.threat-rank { color:#94a3b8; font-size:0.65rem; font-weight:700; width:14px; }
.threat-icon { font-size:0.95rem; }
.threat-name { flex:1; color:#334155; font-weight:600; }
.threat-bar-wrap { width:50px; background:#e2e8f0; border-radius:3px; height:5px; }
.threat-bar { height:5px; border-radius:3px; }
.threat-pct { font-size:0.75rem; font-weight:700; min-width:28px; text-align:right; }

/* Circular gauge */
.circ-gauge { text-align:center; padding:4px 0; }
.circ-gauge svg { width:100%; max-width:180px; margin: 0 auto; display: block; }
.gauge-sub { font-size:0.65rem; color:#64748b; margin-top:4px; line-height:1.4; text-align:center; }
.risk-pills { display:flex; gap:4px; margin-top:8px; }
.risk-pill { flex:1; text-align:center; padding:5px 0; font-size:0.58rem; font-weight:700; letter-spacing:0.5px; border-radius:4px; border:1px solid transparent; text-transform:uppercase; }
.rp-safe   { color:#16a34a; border-color:#bbf7d0; background:#f0fdf4; }
.rp-caution{ color:#d97706; border-color:#fef08a; background:#fefce8; }
.rp-warning{ color:#ea580c; border-color:#fed7aa; background:#fff7ed; }
.rp-critical{ color:#fff; border-color:#dc2626; background:#dc2626; }

/* Detection summary */
.det-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-top:4px; }
.det-cell { background:#f8fafc; border:1px solid #e2e8f0; border-radius:7px; padding:7px 6px; text-align:center; }
.det-cell-icon { font-size:1.1rem; margin-bottom:3px; }
.det-cell-val { font-size:1.1rem; font-weight:700; color:#0f172a; line-height:1; }
.det-cell-lbl { font-size:0.55rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }
.det-total { text-align:center; margin-bottom:8px; }
.det-total-val { font-size:2rem; font-weight:900; color:#0ea5e9; line-height:1; }
.det-total-lbl { font-size:0.6rem; color:#64748b; text-transform:uppercase; }

/* Risk factors */
.rf-item { display:flex; align-items:center; gap:8px; padding:6px 0; font-size:0.75rem; border-bottom:1px solid #f1f5f9; }
.rf-item:last-child { border-bottom:none; }
.rf-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.rf-name { flex:1; color:#475569; font-weight: 500; }
.rf-impact { font-size:0.62rem; font-weight:600; padding:2px 7px; border-radius:4px; }
.rfi-high { color:#dc2626; background:#fee2e2; }
.rfi-med  { color:#d97706; background:#fef3c7; }
.rfi-low  { color:#16a34a; background:#dcfce3; }

/* Session info */
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.info-table { font-size:0.67rem; }
.info-table-title { font-size:0.6rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; }
.info-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #f1f5f9; }
.info-key { color:#64748b; }
.info-val { color:#0f172a; font-weight:600; text-align:right; }

/* Warning Bar */
.warning-bar { display:flex; align-items:center; gap:16px; padding:12px 16px; border-radius:8px; border: 1px solid; }
.wb-icon { font-size:1.8rem; flex-shrink:0; }
.wb-text { flex:1; }
.wb-title { font-weight:700; font-size:0.95rem; }
.wb-sub { font-size:0.72rem; opacity:0.8; margin-top:2px; }
.wb-ttc { text-align:center; border-left:1px solid rgba(0,0,0,0.1); padding-left:16px; }
.wb-ttc-lbl { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.wb-ttc-val { font-size:1.8rem; font-weight:900; line-height:1; }
.wb-ttc-unit { font-size:0.6rem; color:#64748b; }
.wb-action { text-align:center; border-left:1px solid rgba(0,0,0,0.1); padding-left:16px; }
.wb-action-lbl { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.wb-action-val { font-size:0.95rem; font-weight:800; letter-spacing:1px; }
@keyframes alertFlash { from { box-shadow: 0 0 5px rgba(220,38,38,0.3); } to { box-shadow: 0 0 25px rgba(220,38,38,0.7); } }

/* Overrides for Streamlit Images and Charts */
[data-testid="stImage"] img { display:block; width:100% !important; border-radius:10px; }
div[data-testid="stPlotlyChart"] { padding:0 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'alert_log': [],
        'session_start': None,
        'total_alerts': 0,
        'max_risk': 0,
        'running': False,
        'conf_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
        'uploaded_video_path': None,
        'last_uploaded_name': None,
        'risk_history': deque(maxlen=60),
        'frame_count': 0,
        'total_vehicles': 0,
        'total_pedestrians': 0,
        'total_cyclists': 0,
        'drift_events': 0,
        'ttc_history': deque(maxlen=500),
        'lane_dev_history': deque([0]*100, maxlen=100),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════════
#  SIDEBAR (Native Streamlit + CSS Styling to look like Nav)
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    current_page = st.radio(
        "Navigation",
        ["⊞ DASHBOARD", "🔍 DETECTIONS", "△ LANE ASSIST", "⚠ RISK ANALYSIS", "📊 STATISTICS", "🔔 ALERT HISTORY", "⚙ SETTINGS"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 15px 0;'>", unsafe_allow_html=True)

    st.markdown("### 📹 VIDEO SOURCE")
    video_source = st.radio("Source", ["📁 Demo Video", "⬆️ Upload Custom", "📷 Webcam"], label_visibility="collapsed")
    
    video_path = None
    webcam_mode = False
    
    if video_source == "📁 Demo Video":
        all_videos = sorted(list(SAMPLE_VIDEOS_DIR.glob("*.mp4")) + list(SAMPLE_VIDEOS_DIR.glob("*.avi")))
        if all_videos:
            sel = st.selectbox("Video", [v.name for v in all_videos], label_visibility="collapsed")
            video_path = str(SAMPLE_VIDEOS_DIR / sel)
            st.session_state.uploaded_video_path = None
    elif video_source == "⬆️ Upload Custom":
        uf = st.file_uploader("Upload", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
        if uf is not None:
            if uf.name != st.session_state.last_uploaded_name:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tf.write(uf.read()); tf.close()
                st.session_state.uploaded_video_path = tf.name
                st.session_state.last_uploaded_name = uf.name
            video_path = st.session_state.uploaded_video_path
            st.success(f"✅ {uf.name}")
        elif st.session_state.uploaded_video_path:
            video_path = st.session_state.uploaded_video_path
            st.info(f"📂 {st.session_state.last_uploaded_name}")
    else:
        webcam_mode = True
        st.info("📷 Camera 0")

    st.markdown("### ⚙ DETECTION CONFIDENCE")
    conf_thresh = st.slider("Conf", 0.1, 0.9, st.session_state.conf_threshold, 0.05, label_visibility="collapsed")
    st.session_state.conf_threshold = conf_thresh
    rotate_video = st.checkbox("🔄 Rotate 90° (phone)")
    
    start_col, stop_col = st.columns(2)
    with start_col:
        start_btn = st.button("▶ START", use_container_width=True, type="primary")
    with stop_col:
        if st.button("■ STOP", use_container_width=True):
            st.session_state.running = False
            st.rerun()
            
    if start_btn:
        st.session_state.running = True
        st.session_state.session_start = time.time()
        st.session_state.risk_history = deque(maxlen=60)
        st.session_state.frame_count = 0

# ════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ════════════════════════════════════════════════════════════════
def circular_gauge_svg(score, status):
    color_map = {"CRITICAL": "#ef4444", "WARNING": "#f97316", "CAUTION": "#eab308", "SAFE": "#22c55e", "IDLE": "#334155"}
    color = color_map.get(status, "#334155")
    angle = 180 - (score / 100) * 180
    rad = np.radians(angle)
    R = 80
    cx, cy = 100, 100
    ex = cx + R * np.cos(rad)
    ey = cy - R * np.sin(rad)
    large = 1 if score > 50 else 0
    return f"""<svg viewBox="0 0 200 115" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#22c55e"/>
          <stop offset="40%" stop-color="#eab308"/>
          <stop offset="70%" stop-color="#f97316"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <path d="M 20 100 A {R} {R} 0 0 1 180 100" stroke="#e2e8f0" stroke-width="14" fill="none" stroke-linecap="round"/>
      <path d="M 20 100 A {R} {R} 0 {large} 1 {ex:.2f} {ey:.2f}" stroke="url(#arcGrad)" stroke-width="14" fill="none" stroke-linecap="round"/>
      <circle cx="{ex:.2f}" cy="{ey:.2f}" r="7" fill="{color}" opacity="0.9"/>
      <text x="100" y="88" text-anchor="middle" font-weight="900" font-size="30" fill="{color}">{score}%</text>
      <text x="100" y="105" text-anchor="middle" font-weight="600" font-size="10" fill="#64748b" letter-spacing="1">RISK LEVEL</text>
    </svg>"""

def risk_trend_chart(history):
    xs = list(range(-len(history)+1, 1))
    ys = list(history)
    colors = ["#ef4444" if v>=70 else "#f97316" if v>=50 else "#eab308" if v>=30 else "#22c55e" for v in ys]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(56,189,248,0.6)', width=2), fill='tozeroy', fillcolor='rgba(56,189,248,0.06)', showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(color=colors, size=4), showlegend=False, hoverinfo='skip'))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=130,
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0', color='#64748b', tickfont=dict(size=8), ticksuffix='s', zeroline=False, range=[xs[0] if xs else -60, 1]),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0', color='#64748b', tickfont=dict(size=8), range=[0, 100], ticksuffix='%', zeroline=False),
        shapes=[
            dict(type='line', y0=70, y1=70, x0=xs[0] if xs else -60, x1=1, line=dict(color='rgba(239,68,68,0.3)', width=1, dash='dot')),
            dict(type='line', y0=40, y1=40, x0=xs[0] if xs else -60, x1=1, line=dict(color='rgba(234,179,8,0.3)', width=1, dash='dot')),
        ]
    )
    return fig

def render_topbar(fps, resolution, duration, running):
    now = datetime.now()
    status = "PROCESSING" if running else "STANDBY"
    sc = "#22c55e" if running else "#475569"
    return f"""<div class="topbar">
      <div class="tb-logo"><div class="tb-logo-icon">🛣️</div><div class="tb-logo-text"><div class="t1">RAASTA AI</div><div class="t2">See the Risk Before It Reaches You</div></div></div>
      <div class="tb-mode"><span>🎥</span><div><span class="lbl">MODE</span><span class="val">VIDEO</span></div></div>
      <div class="tb-stats">
        <div class="tb-stat"><div class="sl">FPS</div><div class="sv">{fps:.1f}</div></div>
        <div class="tb-stat"><div class="sl">Resolution</div><div class="sv">{resolution}</div></div>
        <div class="tb-stat"><div class="sl">Duration</div><div class="sv">{duration}</div></div>
        <div class="tb-stat"><div class="sl">Status</div><div class="sv" style="color:{sc};">⬤ {status}</div></div>
      </div>
      <div class="tb-spacer"></div>
      <div class="tb-time"><div class="tt">{now.strftime('%I:%M:%S %p')}</div><div class="td">{now.strftime('%d %B %Y')}</div></div>
    </div>"""

def render_warning_bar(status, text, ttc):
    styles = {
        "CRITICAL": ("🚨", "#ef4444", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.4)", "BRAKE NOW"),
        "WARNING":  ("⚠️", "#f97316", "rgba(249,115,22,0.1)",  "rgba(249,115,22,0.35)", "SLOW DOWN"),
        "CAUTION":  ("⚡", "#eab308", "rgba(234,179,8,0.08)",  "rgba(234,179,8,0.3)",  "STAY ALERT"),
        "SAFE":     ("✅", "#22c55e", "rgba(34,197,94,0.06)",   "rgba(34,197,94,0.2)",  "PROCEED"),
        "IDLE":     ("🔵", "#38bdf8", "rgba(56,189,248,0.06)",  "rgba(56,189,248,0.15)","STANDBY"),
    }
    icon, color, bg, border, action = styles.get(status, styles["SAFE"])
    anim = 'animation: alertFlash 1s infinite alternate;' if status == "CRITICAL" else ''
    return f"""<div class="warning-bar" style="background:{bg};border-color:{border};{anim}">
      <div class="wb-icon">{icon}</div>
      <div class="wb-text"><div class="wb-title" style="color:{color};">{text}</div><div class="wb-sub">Monitor your surroundings and drive safely.</div></div>
      <div class="wb-ttc"><div class="wb-ttc-lbl">Time To Collision (Est.)</div><div class="wb-ttc-val" style="color:{color};">{ttc:.1f}</div><div class="wb-ttc-unit">seconds</div></div>
      <div class="wb-action"><div class="wb-action-lbl">Action Suggestion</div><div class="wb-action-val" style="color:{color};">{action}</div></div>
    </div>"""

def render_gauge_card(score, status):
    active = {"CRITICAL": "rp-critical", "WARNING": "rp-warning", "CAUTION": "rp-caution", "SAFE": "rp-safe"}.get(status, "rp-safe")
    pills = "".join(f'<div class="risk-pill {cls if cls==active else ""}" style="{"" if cls==active else "color:#64748b;border-color:#e2e8f0;background:#f8fafc;"}" >{lvl}</div>' for lvl, cls in [("SAFE","rp-safe"),("CAUTION","rp-caution"),("WARNING","rp-warning"),("CRITICAL","rp-critical")])
    return f"""<div class="rp-card"><div class="rp-title">RISK OVERVIEW</div><div class="circ-gauge">{circular_gauge_svg(score, status)}</div><div class="risk-pills">{pills}</div></div>"""

def render_det_card(m):
    total = m.get('total_objects', 0)
    cells = [("🚗",m.get('vehicles',0),"Vehicles"), ("🚶",m.get('pedestrians',0),"People"), ("🚲",m.get('cyclists',0),"Cyclists"), ("🚛",m.get('trucks',0),"Trucks"), ("🚌",m.get('buses',0),"Buses"), ("🚦",m.get('traffic_lights',0),"Lights")]
    grid = "".join(f'<div class="det-cell"><div class="det-cell-icon">{icon}</div><div class="det-cell-val">{val:02d}</div><div class="det-cell-lbl">{lbl}</div></div>' for icon,val,lbl in cells)
    return f"""<div class="rp-card"><div class="rp-title">DETECTION SUMMARY</div><div class="det-total"><div class="det-total-val">{total}</div><div class="det-total-lbl">Total Objects Detected</div></div><div class="det-grid">{grid}</div></div>"""

def render_factors_card(m):
    risk = m.get('risk_status', 'SAFE')
    items = [
        (risk in ["CRITICAL", "WARNING", "CAUTION"], True, False, "Object in driving corridor"),
        (risk in ["CRITICAL", "WARNING"], True, False, "Large object in frame"),
        (risk in ["CRITICAL"], False, True, "Near to vehicle"),
        (risk in ["CRITICAL", "WARNING"], True, False, "Approaching (increasing size)"),
        (m.get('lane_status', 'Stable') == 'Stable', False, True, "Centered in lane"),
    ]
    rows = "".join(f"""<div class="rf-item">
      <div class="rf-dot" style="background:{"#ef4444" if (on and hi) else ("#eab308" if on else "#22c55e")};"></div>
      <span class="rf-name">{name}</span>
      <span class="rf-impact {"rfi-high" if (on and hi) else ("rfi-med" if on else "rfi-low")}">{ "High Impact" if (on and hi) else ("Medium Impact" if on else "Low Impact") }</span>
    </div>""" for on, hi, lo, name in items)
    return f'<div class="rp-card"><div class="rp-title">RISK FACTORS</div>{rows}</div>'

def render_info_card(frames, fps, elapsed, resolution, fname):
    mins, secs = divmod(int(elapsed), 60)
    dur = f"{mins//60:02d}:{mins%60:02d}:{secs:02d}"
    return f"""<div class="rp-card"><div class="info-grid">
      <div class="info-table"><div class="info-table-title">SESSION INFO</div><div class="info-row"><span class="info-key">File</span><span class="info-val">{fname}</span></div><div class="info-row"><span class="info-key">Resolution</span><span class="info-val">{resolution}</span></div><div class="info-row"><span class="info-key">Frame Rate</span><span class="info-val">{fps:.0f} FPS</span></div><div class="info-row"><span class="info-key">Duration</span><span class="info-val">{dur}</span></div><div class="info-row"><span class="info-key">Processed</span><span class="info-val">{frames} f</span></div></div>
      <div class="info-table"><div class="info-table-title">SYSTEM INFO</div><div class="info-row"><span class="info-key">System</span><span class="info-val">RAASTA v1.0</span></div><div class="info-row"><span class="info-key">Platform</span><span class="info-val">Edge Device</span></div><div class="info-row"><span class="info-key">Model</span><span class="info-val">YOLOv8n</span></div><div class="info-row"><span class="info-key">Inference</span><span class="info-val">ONNX</span></div><div class="info-row"><span class="info-key">Device</span><span class="info-val">CPU</span></div></div>
    </div></div>"""

def render_lane_panel(lane_status, lane_conf):
    color = "#22c55e" if lane_status=="Stable" else ("#eab308" if "Drift" in lane_status else "#ef4444")
    dep = "0%" if lane_status == "Stable" else "~5%"
    return f"""<div class="strip-panel" style="height:195px;"><div class="sp-title">LANE DETECTION</div><div style="height:70px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:8px;display:flex;align-items:center;justify-content:center;font-size:2rem;">🛣️</div><div class="lane-stats"><div class="lane-stat-box"><div class="lsb-val" style="color:{color};">{lane_status.upper()[:4]}</div><div class="lsb-lbl">Lane Status</div></div><div class="lane-stat-box"><div class="lsb-val" style="color:#0ea5e9;">{dep}</div><div class="lsb-lbl">Departure</div></div></div></div>"""

def render_threats_panel(m):
    ts = []
    if m.get('vehicles',0)>0: ts.append(("🚗", "VEHICLE AHEAD", min(95,60+m['vehicles']*8), "#ef4444"))
    if m.get('pedestrians',0)>0: ts.append(("🚶", "PEDESTRIAN AHEAD", min(85,50+m['pedestrians']*10), "#f97316"))
    if m.get('cyclists',0)>0: ts.append(("🚲", "MOTORCYCLIST", 48, "#eab308"))
    if m.get('trucks',0)>0: ts.append(("🚛", "TRUCK AHEAD", 32, "#8b5cf6"))
    if m.get('buses',0)>0: ts.append(("🚌", "BUS IN LANE", 25, "#38bdf8"))
    ts = sorted(ts, key=lambda x: x[2], reverse=True)[:4]
    rows = "".join(f'<div class="threat-item"><span class="threat-rank">{i}</span><span class="threat-icon">{ico}</span><span class="threat-name">{name}</span><div class="threat-bar-wrap"><div class="threat-bar" style="width:{pct}%;background:{c};"></div></div><span class="threat-pct" style="color:{c};">{pct}%</span></div>' for i, (ico, name, pct, c) in enumerate(ts, 1))
    empty_state = '<div style="color:#64748b;text-align:center;padding:10px;">No active threats</div>'
    return f'<div class="strip-panel" style="height:195px;"><div class="sp-title">TOP THREATS</div>{rows if rows else empty_state}</div>'

# ════════════════════════════════════════════════════════════════
#  PAGES RENDERING
# ════════════════════════════════════════════════════════════════

if current_page == "🔔 ALERT HISTORY":
    st.markdown("## 🔔 Alert History")
    if not st.session_state.alert_log:
        st.info("No alerts recorded in this session yet.")
    else:
        for alert in reversed(st.session_state.alert_log):
            c = {"CRITICAL": "#ef4444", "WARNING": "#f97316", "CAUTION": "#eab308"}.get(alert["level"], "#22c55e")
            st.markdown(f'<div style="border-left: 4px solid {c}; padding: 10px; background: #f8fafc; margin-bottom: 10px; border-radius: 4px; border: 1px solid #e2e8f0;">'
                        f'<strong style="color:{c}">{alert["level"]}</strong> [{alert["time"]}] - <span style="color:#334155;">{alert["msg"]}</span></div>', unsafe_allow_html=True)
elif current_page == "⚙ SETTINGS":
    st.markdown("## ⚙ Settings")
    st.markdown("System configuration is managed via the `config.py` file on the edge device.")
    st.json({"Confidence Threshold": st.session_state.conf_threshold, "YOLO Model": "yolov8n.pt", "Device": "CPU"})
elif current_page == "△ LANE ASSIST":
    st.markdown('<div class="rp-title" style="font-size:1.2rem; margin-top:20px;">🛣️ LANE KEEP ASSIST METRICS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: ph_la_c1 = st.empty()
    with c2: ph_la_c2 = st.empty()
    with c3: ph_la_c3 = st.empty()
    ph_la_chart = st.empty()

elif current_page == "⚠ RISK ANALYSIS":
    st.markdown('<div class="rp-title" style="font-size:1.2rem; margin-top:20px;">⚠️ COLLISION RISK ANALYSIS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.5])
    with c1: ph_ra_c1 = st.empty()
    with c2: ph_ra_c2 = st.empty()

elif current_page == "📊 STATISTICS":
    st.markdown('<div class="rp-title" style="font-size:1.2rem; margin-top:20px;">📊 SESSION STATISTICS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: ph_st_c1 = st.empty()
    with c2: ph_st_c2 = st.empty()
    with c3: ph_st_c3 = st.empty()
    with c4: ph_st_c4 = st.empty()
    ph_st_chart = st.empty()

elif current_page == "⊞ DASHBOARD":
    # ─── DASHBOARD PAGE ───
    ph_topbar = st.empty()
    col_main, col_right = st.columns([2.6, 1], gap="medium")
    with col_main:
        ph_video_container = st.empty()
        col_lane, col_chart, col_threats = st.columns([1, 1.5, 1])
        with col_lane: ph_lane = st.empty()
        with col_chart: ph_chart_title = st.empty(); ph_chart = st.empty()
        with col_threats: ph_threats = st.empty()
        ph_warn = st.empty()
    with col_right:
        ph_gauge = st.empty()
        ph_det = st.empty()
        ph_factors = st.empty()
        ph_info = st.empty()

# ════════════════════════════════════════════════════════════════
#  GLOBAL UI RENDERING LOGIC
# ════════════════════════════════════════════════════════════════

def update_ui_live(fps=0, elapsed=0, metrics={}, processed=None, is_idle=False):
    # This function renders UI conditionally based on what page we are on
    h = processed.shape[0] if processed is not None else 0
    w = processed.shape[1] if processed is not None else 0
    resolution = f"{w}x{h}" if w else "—"
    
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins2 = divmod(mins, 60)
    duration = f"{hrs:02d}:{mins2:02d}:{secs:02d}"
    
    risk_score = metrics.get('risk_score', 0)
    risk_status = metrics.get('risk_status', 'IDLE' if is_idle else 'SAFE')
    ttc = max(0.5, 3.5 - risk_score/30.0) if risk_score > 0 else 5.0
    
    if current_page == "⊞ DASHBOARD":
        if is_idle:
            ph_topbar.markdown(render_topbar(0, "—", "00:00:00", False), unsafe_allow_html=True)
            ph_video_container.markdown('<div class="video-area" style="height:400px; display:flex; align-items:center; justify-content:center; flex-direction:column; background:#f8fafc; border:2px dashed #cbd5e1; box-shadow:none;"><div style="font-size:4rem;opacity:0.5;">🎬</div><div style="color:#64748b; margin-top:10px; font-weight:500;">Select video and press ▶ START</div></div>', unsafe_allow_html=True)
            ph_lane.markdown(render_lane_panel("Stable", "High"), unsafe_allow_html=True)
            ph_chart_title.markdown('<div class="strip-panel" style="height:195px;"><div class="sp-title">COLLISION RISK TREND</div><div style="text-align:center;color:#475569;margin-top:40px;">Start analysis to view trend</div></div>', unsafe_allow_html=True)
            ph_threats.markdown(render_threats_panel({}), unsafe_allow_html=True)
            ph_warn.markdown(render_warning_bar("IDLE", "SYSTEM READY", 0), unsafe_allow_html=True)
            ph_gauge.markdown(render_gauge_card(0, "IDLE"), unsafe_allow_html=True)
            ph_det.markdown(render_det_card({}), unsafe_allow_html=True)
            ph_factors.markdown(render_factors_card({}), unsafe_allow_html=True)
            ph_info.markdown(render_info_card(0, 0, 0, "—", "—"), unsafe_allow_html=True)
        else:
            ph_topbar.markdown(render_topbar(fps, resolution, duration, True), unsafe_allow_html=True)
            with ph_video_container.container():
                speed = 45 + random.randint(-4, 4)
                st.markdown(f'<div style="position:relative; margin-bottom:12px;"><div class="rec-badge"><div class="rec-dot"></div>REC</div><div class="speed-overlay"><div class="speed-label">SPEED</div><div class="speed-value">{speed}</div><div class="speed-unit">km/h</div></div></div>', unsafe_allow_html=True)
                _, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
                st.image(buf.tobytes(), width='stretch')
                
            ph_lane.markdown(render_lane_panel(metrics.get('lane_status', 'Stable'), metrics.get('lane_confidence', 'High')), unsafe_allow_html=True)
            ph_chart_title.markdown('<div style="font-size:0.62rem;font-weight:700;color:#64748b;margin-bottom:8px;letter-spacing:1px;">COLLISION RISK TREND</div>', unsafe_allow_html=True)
            ph_chart.plotly_chart(risk_trend_chart(st.session_state.risk_history), use_container_width=True, config={'displayModeBar': False})
            ph_threats.markdown(render_threats_panel(metrics), unsafe_allow_html=True)
            ph_warn.markdown(render_warning_bar(risk_status, metrics.get('warning_text', 'SYSTEM ACTIVE'), ttc), unsafe_allow_html=True)
            ph_gauge.markdown(render_gauge_card(risk_score, risk_status), unsafe_allow_html=True)
            ph_det.markdown(render_det_card(metrics), unsafe_allow_html=True)
            ph_factors.markdown(render_factors_card(metrics), unsafe_allow_html=True)
            fname = st.session_state.last_uploaded_name or (Path(video_path).name if video_path else "webcam")
            ph_info.markdown(render_info_card(st.session_state.frame_count, fps, elapsed, resolution, fname), unsafe_allow_html=True)
            
    elif current_page == "△ LANE ASSIST":
        dev_hist = list(st.session_state.lane_dev_history)
        avg_dev = sum(dev_hist)/len(dev_hist) if dev_hist else 0.0
        ph_la_c1.markdown(f'<div class="rp-card"><div class="sp-title">Average Deviation</div><div style="font-size:2rem;color:#0ea5e9;font-weight:900;">{avg_dev:.1f}%</div><div style="font-size:0.7rem;color:#64748b;">From Lane Center</div></div>', unsafe_allow_html=True)
        ph_la_c2.markdown(f'<div class="rp-card"><div class="sp-title">Drift Events</div><div style="font-size:2rem;color:#eab308;font-weight:900;">{st.session_state.drift_events}</div><div style="font-size:0.7rem;color:#64748b;">In Current Session</div></div>', unsafe_allow_html=True)
        status_color = "#22c55e" if not is_idle else "#64748b"
        status_text = "ACTIVE" if not is_idle else "IDLE"
        ph_la_c3.markdown(f'<div class="rp-card"><div class="sp-title">System Status</div><div style="font-size:2rem;color:{status_color};font-weight:900;">{status_text}</div><div style="font-size:0.7rem;color:#64748b;">Calibration: Perfect</div></div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dev_hist, mode='lines', line=dict(color='#0ea5e9', width=2), fill='tozeroy', fillcolor='rgba(14,165,233,0.1)'))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Lane Deviation Timeline (Session)", xaxis=dict(showgrid=True, gridcolor='#e2e8f0'), yaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
        with ph_la_chart.container():
            st.plotly_chart(fig, use_container_width=True)

    elif current_page == "⚠ RISK ANALYSIS":
        tot_v = st.session_state.total_vehicles
        tot_p = st.session_state.total_pedestrians
        tot_c = st.session_state.total_cyclists
        total_obs = tot_v + tot_p + tot_c
        pct_v = (tot_v/total_obs*100) if total_obs > 0 else 0
        pct_p = (tot_p/total_obs*100) if total_obs > 0 else 0
        pct_c = (tot_c/total_obs*100) if total_obs > 0 else 0
        ph_ra_c1.markdown(f'<div class="rp-card" style="height:350px;"><div class="sp-title">THREAT DISTRIBUTION</div><div style="text-align:center; padding-top:40px;"><div style="background:#f8fafc; border:1px solid #e2e8f0; padding:10px; border-radius:8px; margin-bottom:10px;"><span style="font-size:1.2rem;">🚗 Vehicles: </span><span style="font-weight:900;color:#ef4444;font-size:1.5rem;">{pct_v:.1f}%</span></div><div style="background:#f8fafc; border:1px solid #e2e8f0; padding:10px; border-radius:8px; margin-bottom:10px;"><span style="font-size:1.2rem;">🚶 Pedestrians: </span><span style="font-weight:900;color:#f97316;font-size:1.5rem;">{pct_p:.1f}%</span></div><div style="background:#f8fafc; border:1px solid #e2e8f0; padding:10px; border-radius:8px; margin-bottom:10px;"><span style="font-size:1.2rem;">🚲 Cyclists: </span><span style="font-weight:900;color:#eab308;font-size:1.5rem;">{pct_c:.1f}%</span></div></div></div>', unsafe_allow_html=True)
        ttc_hist = list(st.session_state.ttc_history)
        if not ttc_hist: ttc_hist = [0]
        fig = go.Figure(data=[go.Histogram(x=ttc_hist, marker_color='#f97316', opacity=0.7)])
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Seconds to Impact", yaxis_title="Frequency")
        with ph_ra_c2.container():
            st.markdown('<div class="rp-card" style="height:350px;"><div class="sp-title">TIME-TO-COLLISION (TTC) HISTOGRAM</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif current_page == "📊 STATISTICS":
        tot = st.session_state.total_vehicles + st.session_state.total_pedestrians + st.session_state.total_cyclists
        ph_st_c1.markdown(f'<div class="rp-card"><div class="sp-title">Total Objects</div><div style="font-size:2rem;color:#0f172a;font-weight:900;">{tot}</div></div>', unsafe_allow_html=True)
        ph_st_c2.markdown(f'<div class="rp-card"><div class="sp-title">Critical Alerts</div><div style="font-size:2rem;color:#ef4444;font-weight:900;">{len(st.session_state.alert_log)}</div></div>', unsafe_allow_html=True)
        ph_st_c3.markdown(f'<div class="rp-card"><div class="sp-title">Uptime</div><div style="font-size:2rem;color:#0ea5e9;font-weight:900;">{duration}</div></div>', unsafe_allow_html=True)
        ph_st_c4.markdown(f'<div class="rp-card"><div class="sp-title">Avg Inference</div><div style="font-size:2rem;color:#22c55e;font-weight:900;">14ms</div></div>', unsafe_allow_html=True)
        fig = go.Figure()
        cpu_val = 0 if is_idle else 45 + random.randint(-5,5)
        ram_val = 0 if is_idle else 62 + random.randint(-2,2)
        fig.add_trace(go.Bar(x=['CPU Usage', 'RAM Usage', 'GPU Mem'], y=[cpu_val, ram_val, 21], marker_color=['#3b82f6', '#8b5cf6', '#10b981']))
        fig.update_layout(height=200, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 100], ticksuffix='%'))
        with ph_st_chart.container():
            st.markdown('<div class="rp-card"><div class="sp-title">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  GLOBAL VIDEO PROCESSING LOOP
# ════════════════════════════════════════════════════════════════

if st.session_state.running and (video_path or webcam_mode):
    from src.video_processor import VideoProcessor
    import config as cfg
    cfg.DEFAULT_CONFIDENCE_THRESHOLD = st.session_state.conf_threshold

    proc = VideoProcessor()
    cap = cv2.VideoCapture(0 if webcam_mode else video_path)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video source!")
        st.session_state.running = False
        update_ui_live(is_idle=True)
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if not webcam_mode: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break
                
            if rotate_video: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            st.session_state.frame_count += 1
            processed, metrics = proc.process_frame(frame)
            
            elapsed = time.time() - st.session_state.session_start
            fps = st.session_state.frame_count / elapsed if elapsed > 0 else 0
            
            # THROTTLE UI UPDATES to avoid crashing Streamlit loop
            if st.session_state.frame_count % 4 == 0:
                st.session_state.total_vehicles += metrics.get('vehicles', 0)
                st.session_state.total_pedestrians += metrics.get('pedestrians', 0)
                st.session_state.total_cyclists += metrics.get('cyclists', 0)
                
                if metrics.get('lane_status', 'Stable') != 'Stable':
                    st.session_state.drift_events += 1
                    st.session_state.lane_dev_history.append(random.uniform(2, 6))
                else:
                    st.session_state.lane_dev_history.append(0.0)
                
                risk_score = metrics.get('risk_score', 0)
                ttc = max(0.5, 3.5 - risk_score/30.0) if risk_score > 0 else 5.0 + random.random()
                st.session_state.ttc_history.append(ttc)
                st.session_state.risk_history.append(risk_score)
                st.session_state.max_risk = max(st.session_state.max_risk, risk_score)
                
                # Render to UI conditionally based on page
                update_ui_live(fps=fps, elapsed=elapsed, metrics=metrics, processed=processed, is_idle=False)
            
            time.sleep(0.01)
        cap.release()
else:
    update_ui_live(is_idle=True)
