# Vehicle Detection and Counting System using YOLOv8

Mini Project - Network & Signal Processing (NSP)
Real-time detection, tracking and direction-wise counting of vehicles in traffic video.

> **Team:** replace the placeholders below with your names and roll numbers.

| Member | Name | Roll No. | Responsibility | Files |
|---|---|---|---|---|
| 1 | _Name 1_ | _..._ | Dataset, YOLO detection, model training | `src/detector.py`, `train.py` |
| 2 | _Name 2_ | _..._ | Tracking and line-crossing counting logic | `src/tracker.py`, `src/counter.py`, `tests/` |
| 3 | _Name 3_ | _..._ | Visualisation, evaluation, charts, report | `src/visualizer.py`, `src/plots.py`, `src/evaluate.py`, `docs/` |

`main.py` (the pipeline) was integrated by all three members.

---

## 1. How it works

```
 Video / Webcam
      |
      v
 [1] YOLOv8 detection  --> cars, buses, trucks, motorcycles, bicycles
      |
      v
 [2] ByteTrack tracking --> unique ID for every vehicle
      |
      v
 [2] Line counter       --> counts each ID once, per class and per direction (IN / OUT)
      |
      v
 [3] Visualiser + Evaluation --> annotated video, events.csv, summary.json, charts
```

**Counting rule:** a vehicle is counted when the bottom-centre of its box moves from one side of
the virtual line to the other. Each track ID is counted only once, so flicker or reversing does not
cause double counting.

## 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows      (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
A GPU is optional. With CPU use `yolov8n.pt` (fastest). With an NVIDIA GPU you can use `yolov8s.pt` or `yolov8m.pt` for better accuracy.
The model weights are downloaded automatically on the first run.

## 3. Run

```bash
python main.py --source traffic.mp4                 # default line at 60% of frame height
python main.py --source traffic.mp4 --select-line   # click two points to place the line
python main.py --source traffic.mp4 --show          # live preview window (q to quit)
python main.py --source 0                           # webcam
python main.py --source traffic.mp4 --weights yolov8s.pt --device 0 --tracker botsort.yaml
```
Useful options: `--conf 0.4` (fewer false detections), `--flip` (swap IN/OUT), `--anchor center`.

**Where to get test videos:** Pexels / Pixabay ("traffic highway top view"), the UA-DETRAC dataset,
or record a clip from a footbridge with your phone. A fixed camera looking down the road works best.

## 4. Output (inside `outputs/<run>/`)

| File | Content |
|---|---|
| `annotated.mp4` | Video with boxes, IDs, trails, counting line and live counter panel |
| `events.csv` | One row per counted vehicle: frame, time, ID, type, direction |
| `summary.json` | Totals, per-class and per-direction counts, average FPS |
| `counts_by_class.png`, `count_timeline.png` | Figures for the report |

## 5. Evaluate (for the report)

1. Count the vehicles in your video **manually** and write them to a JSON file, e.g. `data/ground_truth.json`:
   `{"car": 42, "motorcycle": 31, "bus": 4, "truck": 7}`
2. Compare:
   ```bash
   python -m src.evaluate --summary outputs/<run>/summary.json --gt data/ground_truth.json
   ```
3. Detection metrics (precision, recall, mAP) on a labelled dataset:
   ```bash
   python -m src.evaluate --map --weights yolov8n.pt --data path/to/data.yaml
   ```
4. Run the unit tests: `python -m unittest tests.test_counter -v`

## 6. Improving it (use this for your paper / novelty)

* **Indian traffic classes:** fine-tune with `train.py` on a dataset containing auto-rickshaws and e-rickshaws.
* Compare `yolov8n / s / m` and `bytetrack / botsort` in a table of accuracy versus FPS.
* Test under night, rain and heavy congestion and report where it fails.
* Add speed estimation or lane-wise counting.

## 7. Limitations

* Heavy occlusion in dense traffic can cause ID switches, which leads to over- or under-counting.
* The COCO model has no auto-rickshaw class; such vehicles may be detected as car or motorcycle.
* Accuracy depends on camera angle, so a fixed, elevated camera works best.

## 8. Project structure

```
vehicle_counter/
|-- main.py              pipeline (entry point)
|-- train.py             optional fine-tuning
|-- requirements.txt
|-- src/
|   |-- detector.py      [Member 1] YOLOv8 detection + tracker IDs
|   |-- tracker.py       [Member 2] trajectory history
|   |-- counter.py       [Member 2] line-crossing counter
|   |-- visualizer.py    [Member 3] drawing and live panel
|   |-- plots.py         [Member 3] report charts
|   |-- evaluate.py      [Member 3] count accuracy + mAP
|   `-- utils.py         shared helpers
|-- tests/test_counter.py
|-- data/ground_truth.json
`-- docs/REPORT_GUIDE.md
```

## 9. References

1. Jocher, G. et al., *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
2. Zhang, Y. et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
3. Lin, T.-Y. et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
4. Wen, L. et al., "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking", 2020.
