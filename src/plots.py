"""
MODULE 3b - CHARTS            
-------------------------------------------------
Creates report-ready figures from the counting results.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt


def save_class_bar_chart(counts: dict, path: str):
    classes = sorted(counts)
    ins = [counts[c]["IN"] for c in classes]
    outs = [counts[c]["OUT"] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(classes, ins, label="IN", color="#2a9d8f")
    ax.bar(classes, outs, bottom=ins, label="OUT", color="#e76f51")
    ax.set_title("Vehicle count by type and direction")
    ax.set_ylabel("Vehicles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_timeline(events: list[dict], path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = [e["time_s"] for e in events]
    if times:
        ax.step(times, range(1, len(times) + 1), where="post", color="#264653")
    ax.set_title("Cumulative vehicle count over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vehicles counted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_all(counts, events, out_dir: str):
    save_class_bar_chart(counts, os.path.join(out_dir, "counts_by_class.png"))
    save_timeline(events, os.path.join(out_dir, "count_timeline.png"))
