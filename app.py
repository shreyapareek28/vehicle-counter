
import os
import tempfile
import time

import streamlit as st

st.set_page_config(
    page_title="Velocity - Vehicle Counter",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Velocity: Real-Time Vehicle Detection & Counting")
st.caption("Powered by YOLOv8 + ByteTrack")

# --------------------------------------------------
# IMPORTS AND ERROR HANDLING
# --------------------------------------------------

try:
    import cv2
except Exception as e:
    st.error("OpenCV could not be loaded.")
    st.code(str(e))
    st.info(
        "Check requirements.txt and the Streamlit Cloud "
        "deployment logs for the OpenCV installation error."
    )
    st.stop()

try:
    from src.counter import LineCounter
    from src.detector import VehicleDetector
    from src.tracker import TrackHistory
    from src.utils import parse_line
    from src.visualizer import Visualizer
except Exception as e:
    st.error("A project module could not be imported.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------

st.sidebar.header("Configuration")

conf = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 0.9, 0.35, 0.05
)

iou = st.sidebar.slider(
    "NMS IoU Threshold",
    0.1, 0.9, 0.50, 0.05
)

tracker = st.sidebar.selectbox(
    "Tracker",
    ["bytetrack.yaml", "botsort.yaml"]
)

line_y = st.sidebar.slider(
    "Counting Line Position (Vertical %)",
    0.1, 0.9, 0.60, 0.05
)

anchor_type = st.sidebar.selectbox(
    "Tracking Anchor Point",
    ["bottom", "center"]
)

flip_dir = st.sidebar.checkbox(
    "Swap IN / OUT Direction",
    value=False
)

show_trails = st.sidebar.checkbox(
    "Show Motion Trails",
    value=True
)


# --------------------------------------------------
# VIDEO INPUT
# --------------------------------------------------

st.subheader("1. Select Traffic Video")

uploaded_file = st.file_uploader(
    "Upload an MP4, AVI or MOV video",
    type=["mp4", "avi", "mov"]
)

use_sample = st.checkbox(
    "Use built-in sample video (traffic.mp4)",
    value=True
)

video_path = None
temp_video_path = None

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            temp_video_path = temp_file.name

        video_path = temp_video_path

    except Exception as e:
        st.error("Could not save uploaded video.")
        st.exception(e)

elif use_sample and os.path.exists("traffic.mp4"):

    video_path = "traffic.mp4"

elif use_sample:

    st.warning("Sample video traffic.mp4 was not found.")

if video_path:
    st.success("Video ready for processing.")


# --------------------------------------------------
# VEHICLE DETECTION AND COUNTING
# --------------------------------------------------

if video_path and st.button(
    "▶ Start Vehicle Counter",
    type="primary"
):

    cap = None

    try:

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Could not open the selected video.")
            st.stop()

        ok, first_frame = cap.read()

        if not ok:
            st.error("Could not read the first video frame.")
            st.stop()

        height, width = first_frame.shape[:2]

        src_fps = cap.get(cv2.CAP_PROP_FPS)

        if not src_fps or src_fps <= 0:
            src_fps = 30.0

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Configure counting line
        line_str = (
            f"0.05,{line_y:.2f},"
            f"0.95,{line_y:.2f}"
        )

        p1, p2 = parse_line(
            line_str,
            width,
            height
        )

        # Load detector
        with st.spinner("Loading YOLOv8 model..."):

            detector = VehicleDetector(
                "yolov8n.pt",
                conf,
                iou,
                640,
                None,
                tracker
            )

        history = TrackHistory()

        counter = LineCounter(
            p1,
            p2,
            flip=flip_dir
        )

        viz = Visualizer(
            p1,
            p2,
            show_trails=show_trails
        )

        st.success("YOLOv8 model loaded successfully.")

        # Dashboard layout
        col1, col2 = st.columns([2, 1])

        frame_placeholder = col1.empty()
        progress_bar = col1.progress(0)

        metrics_placeholder = col2.empty()

        frame_idx = 0
        fps_smooth = 0.0
        infer_time = 0.0

        start_time = time.time()

        # --------------------------------------------------
        # FRAME PROCESSING LOOP
        # --------------------------------------------------

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            t0 = time.time()

            detections = detector.detect_and_track(frame)

            dt = time.time() - t0
            infer_time += dt

            inst_fps = 1.0 / max(dt, 1e-6)

            if fps_smooth == 0:
                fps_smooth = inst_fps
            else:
                fps_smooth = (
                    0.9 * fps_smooth
                    + 0.1 * inst_fps
                )

            time_s = frame_idx / src_fps

            # Update vehicle tracking and counting
            for d in detections:

                if anchor_type == "center":
                    point = d.center
                else:
                    point = d.bottom_center

                history.update(
                    d.track_id,
                    point,
                    frame_idx
                )

                counted = counter.update(
                    d.track_id,
                    d.cls_name,
                    point,
                    frame_idx,
                    time_s
                )

                if counted:
                    viz.flash_line()

            history.cleanup(frame_idx)

            # Draw annotations
            annotated = viz.draw(
                frame,
                detections,
                history,
                counter,
                fps_smooth
            )

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            # Update dashboard
            with metrics_placeholder.container():

                st.metric(
                    "Total Vehicles Counted",
                    counter.total
                )

                dir_counts = counter.totals_by_direction()

                c1, c2 = st.columns(2)

                c1.metric(
                    "IN",
                    dir_counts.get("IN", 0)
                )

                c2.metric(
                    "OUT",
                    dir_counts.get("OUT", 0)
                )

                st.divider()

                st.write("### Vehicle Breakdown")

                class_counts = counter.totals_by_class()

                if class_counts:

                    for cls_name, count in sorted(
                        class_counts.items()
                    ):

                        st.write(
                            f"**{cls_name.capitalize()}:** {count}"
                        )

                else:

                    st.write("No vehicles counted yet.")

                st.divider()

                st.metric(
                    "Currently Detected",
                    len(detections)
                )

                st.metric(
                    "Processing FPS",
                    f"{fps_smooth:.2f}"
                )

            frame_idx += 1

            if total_frames > 0:

                progress_bar.progress(
                    min(frame_idx / total_frames, 1.0)
                )

        # --------------------------------------------------
        # FINAL RESULTS
        # --------------------------------------------------

        wall_time = time.time() - start_time

        progress_bar.progress(1.0)

        st.success(
            f"Processing completed! "
            f"Total vehicles counted: {counter.total}"
        )

        st.subheader("Final Results")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Total Vehicles",
            counter.total
        )

        m2.metric(
            "Frames Processed",
            frame_idx
        )

        m3.metric(
            "Average Inference FPS",
            f"{frame_idx / infer_time:.2f}"
            if infer_time > 0 else "0"
        )

        st.subheader("Counts by Vehicle Type")

        final_counts = counter.totals_by_class()

        if final_counts:

            st.bar_chart(final_counts)

        else:

            st.warning(
                "No vehicles crossed the counting line. "
                "Check the video, tracking anchor and line position."
            )

        st.subheader("Counts by Direction")

        st.json(counter.totals_by_direction())

        st.caption(
            f"Total processing time: {wall_time:.2f} seconds"
        )

        # Show event records
        if counter.events:

            st.subheader("Vehicle Counting Events")

            st.dataframe(
                counter.events,
                use_container_width=True
            )

    except Exception as e:

        st.error("An error occurred during video processing.")
        st.exception(e)

    finally:

        if cap is not None:
            cap.release()

        if temp_video_path and os.path.exists(temp_video_path):

            try:
                os.remove(temp_video_path)
            except OSError:
                pass
