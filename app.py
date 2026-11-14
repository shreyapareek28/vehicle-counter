import os
import tempfile
import time
import cv2
import streamlit as st

from src.counter import LineCounter
from src.detector import VehicleDetector
from src.tracker import TrackHistory
from src.utils import parse_line
from src.visualizer import Visualizer

st.set_page_config(page_title="Velocity - Vehicle Counter", layout="wide")
st.title("🚗 Velocity: Real-Time Vehicle Detection & Counting")
st.caption("Powered by YOLOv8 + ByteTrack")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("NMS IoU Threshold", 0.1, 0.9, 0.50, 0.05)
tracker = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
line_y = st.sidebar.slider("Counting Line Position (Vertical %)", 0.1, 0.9, 0.42, 0.02)
flip_dir = st.sidebar.checkbox("Swap IN / OUT Direction", value=False)
show_trails = st.sidebar.checkbox("Show Motion Trails", value=True)

# --- Video Input ---
st.subheader("1. Select Traffic Video")
uploaded_file = st.file_uploader("Upload an MP4/AVI video", type=["mp4", "avi", "mov"])
use_sample = st.checkbox("Use built-in sample video (traffic.mp4)", value=True)

video_path = None
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
elif use_sample and os.path.exists("traffic.mp4"):
    video_path = "traffic.mp4"

# --- Run Detection & Counting ---
if video_path and st.button("▶ Start Vehicle Counter", type="primary"):
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    if not ok:
        st.error("Could not read video file.")
        st.stop()

    h, w = first.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    line_str = f"0.05,{line_y:.2f},0.95,{line_y:.2f}"
    p1, p2 = parse_line(line_str, w, h)

    with st.spinner("Loading YOLOv8s model..."):
        detector = VehicleDetector("yolov8s.pt", conf, iou, 640, None, tracker)
    history = TrackHistory()
    counter = LineCounter(p1, p2, flip=flip_dir)
    viz = Visualizer(p1, p2, show_trails=show_trails)

    # UI Layout: Live Video on Left, Live Metrics on Right
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    progress_bar = col1.progress(0)
    metrics_placeholder = col2.empty()

    frame_idx, fps_smooth = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        detections = detector.detect_and_track(frame)
        dt = time.time() - t0
        inst_fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

        time_s = frame_idx / src_fps
        for d in detections:
            point = d.bottom_center
            history.update(d.track_id, point, frame_idx)
            if counter.update(d.track_id, d.cls_name, point, frame_idx, time_s):
                viz.flash_line()
        history.cleanup(frame_idx)

        annotated = viz.draw(frame, detections, history, counter, fps_smooth)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Live Dashboard
        with metrics_placeholder.container():
            st.metric("Total Vehicles Counted", counter.total)
            dir_counts = counter.totals_by_direction()
            c1, c2 = st.columns(2)
            c1.metric("IN", dir_counts.get("IN", 0))
            c2.metric("OUT", dir_counts.get("OUT", 0))
            st.divider()
            st.write("**Breakdown by Vehicle Type:**")
            for cls_name, count in sorted(counter.totals_by_class().items()):
                st.write(f"- **{cls_name.capitalize()}**: {count}")

        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    st.success(f"Completed! Counted {counter.total} vehicles.")
