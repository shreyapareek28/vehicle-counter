"""
MODULE 3c - EVALUATION          
-------------------------------------------------
Two ways to measure the system, both used in the report:

 1. COUNT ACCURACY  - compare the automatic counts with a manual count
      python -m src.evaluate --summary outputs/<run>/summary.json --gt data/ground_truth.json

 2. DETECTION METRICS - precision / recall / mAP on a labelled dataset
      python -m src.evaluate --map --weights yolov8n.pt --data path/to/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def compare_counts(pred: dict, gt: dict) -> list[dict]:
    rows = []
    for cls in sorted(set(pred) | set(gt)):
        p, g = pred.get(cls, 0), gt.get(cls, 0)
        err = p - g
        acc = 100.0 * (1 - abs(err) / g) if g else (100.0 if p == 0 else 0.0)
        rows.append({"class": cls, "ground_truth": g, "predicted": p,
                     "error": err, "abs_error": abs(err), "accuracy_pct": round(max(acc, 0), 1)})
    total_g, total_p = sum(gt.values()), sum(pred.values())
    total_acc = 100.0 * (1 - abs(total_p - total_g) / total_g) if total_g else 0.0
    rows.append({"class": "TOTAL", "ground_truth": total_g, "predicted": total_p,
                 "error": total_p - total_g, "abs_error": abs(total_p - total_g),
                 "accuracy_pct": round(max(total_acc, 0), 1)})
    return rows


def print_table(rows):
    head = f"{'Class':<14}{'Manual':>8}{'System':>8}{'Error':>7}{'Acc %':>8}"
    print(head + "\n" + "-" * len(head))
    for r in rows:
        print(f"{r['class']:<14}{r['ground_truth']:>8}{r['predicted']:>8}"
              f"{r['error']:>7}{r['accuracy_pct']:>8}")


def run_map(weights: str, data: str):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data)
    print(f"\nPrecision : {m.box.mp:.3f}\nRecall    : {m.box.mr:.3f}"
          f"\nmAP@0.5   : {m.box.map50:.3f}\nmAP@0.5:95: {m.box.map:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the vehicle counting system")
    ap.add_argument("--summary", help="summary.json produced by main.py")
    ap.add_argument("--gt", help="ground-truth JSON, e.g. {\"car\": 40, \"bus\": 3}")
    ap.add_argument("--map", action="store_true", help="run detection metrics instead")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", help="YOLO data.yaml for --map")
    a = ap.parse_args()

    if a.map:
        if not a.data:
            ap.error("--map needs --data")
        return run_map(a.weights, a.data)

    if not (a.summary and a.gt):
        ap.error("give --summary and --gt (or use --map)")
    pred = json.load(open(a.summary))["counts_by_class"]
    gt = json.load(open(a.gt))
    rows = compare_counts(pred, gt)
    print_table(rows)
    out = os.path.join(os.path.dirname(a.summary), "evaluation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
