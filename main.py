"""
Vehicle Detection and Counting System using YOLOv8
==================================================
Run:  python main.py --source traffic.mp4
      python main.py --source traffic.mp4 --select-line --show
      python main.py --source 0                      (webcam)

Pipeline:  video -> YOLOv8 detection -> ByteTrack IDs -> line counter
           -> annotated video + events.csv + summary.json + charts
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2

from src.counter import LineCounter
from src.detector import VehicleDetector
from src.plots import save_all
from src.tracker import TrackHistory
from src.utils import make_run_dir, parse_line
from src.visualizer import Visualizer


def get_args():
    ap = argparse.ArgumentParser(description="Vehicle detection & counting (YOLOv8 + ByteTrack)")
    ap.add_argument("--source", required=True, help="video path or webcam index (0)")
    ap.add_argument("--weights", default="yolov8n.pt", help="yolov8n/s/m.pt or your custom .pt")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--device", default=None, help="'cpu', '0' (GPU) ... default = auto")
    ap.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--line", default="0.05,0.60,0.95,0.60",
                    help="counting line as fractions x1,y1,x2,y2 of the frame")
    ap.add_argument("--select-line", action="store_true", help="click two points to draw the line")
    ap.add_argument("--anchor", default="bottom", choices=["bottom", "center"],
                    help="which point of the box is tested against the line")
    ap.add_argument("--flip", action="store_true", help="swap IN / OUT labels")
    ap.add_argument("--no-trails", action="store_true")
    ap.add_argument("--show", action="store_true", help="show live window (press q to quit)")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--name", default=None, help="run name (default: timestamp)")
    return ap.parse_args()


def pick_line(frame):
    """Let the user click two points on the first frame."""
    pts, win = [], "Click 2 points for the counting line (ENTER to confirm)"
    img = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y)); cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)

    cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, img)
        if cv2.waitKey(20) & 0xFF in (13, 10) and len(pts) == 2:
            break
    cv2.destroyWindow(win)
    return pts[0], pts[1]


def main():
    args = get_args()
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    h, w = first.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not is_cam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    p1, p2 = pick_line(first) if args.select_line else parse_line(args.line, w, h)

    detector = VehicleDetector(args.weights, args.conf, args.iou, args.imgsz,
                               args.device, args.tracker)
    history = TrackHistory()
    counter = LineCounter(p1, p2, flip=args.flip)
    viz = Visualizer(p1, p2, show_trails=not args.no_trails)

    run_dir = make_run_dir(args.out, args.name)
    writer = cv2.VideoWriter(os.path.join(run_dir, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    print(f"[INFO] classes tracked : {detector.class_names}")
    print(f"[INFO] results folder  : {run_dir}")

    frame_idx, infer_time, fps_smooth = 0, 0.0, 0.0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        detections = detector.detect_and_track(frame)
        dt = time.time() - t0
        infer_time += dt
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6)) if fps_smooth else 1.0 / max(dt, 1e-6)

        time_s = frame_idx / src_fps
        for d in detections:
            point = d.bottom_center if args.anchor == "bottom" else d.center
            history.update(d.track_id, point, frame_idx)
            if counter.update(d.track_id, d.cls_name, point, frame_idx, time_s):
                viz.flash_line()
        history.cleanup(frame_idx)

        annotated = viz.draw(frame, detections, history, counter, fps_smooth)
        writer.write(annotated)
        if args.show:
            cv2.imshow("Vehicle Counter (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[INFO] frame {frame_idx}  counted so far: {counter.total}")

    cap.release(); writer.release(); cv2.destroyAllWindows()
    wall = time.time() - t_start

    # ---- save results -------------------------------------------------
    with open(os.path.join(run_dir, "events.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "vehicle_type", "direction"])
        wr.writeheader(); wr.writerows(counter.events)

    summary = {
        "source": args.source, "weights": args.weights, "tracker": args.tracker,
        "frames_processed": frame_idx, "video_fps": round(src_fps, 2),
        "avg_inference_fps": round(frame_idx / infer_time, 2) if infer_time else 0,
        "wall_time_s": round(wall, 1),
        "total_count": counter.total,
        "counts_by_direction": counter.totals_by_direction(),
        "counts_by_class": counter.totals_by_class(),
        "counts_detail": {c: d for c, d in counter.counts.items()},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    save_all(counter.counts, counter.events, run_dir)

    print("\n========== RESULT ==========")
    print(f"Total vehicles counted : {counter.total}")
    for c, n in sorted(counter.totals_by_class().items()):
        print(f"  {c:<12}: {n}")
    print(f"Average inference FPS  : {summary['avg_inference_fps']}")
    print(f"Saved to               : {run_dir}")


if __name__ == "__main__":
    main()
