#!/usr/bin/env python3
"""
Phase A1 visualization: overlay YOLO bboxes, InceptionV3 labels, and decisions on KITTI frames.
Outputs annotated images → ffmpeg renders video.
"""

import json, os, sys, subprocess
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

SUMMARY = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU/phase_a1_board_summary.json")
KITTI_DIR = os.path.expanduser("~/Documents/端测/测试数据/行车记录仪帧")
OUT_DIR = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU-可视化")
OUT_VIDEO = os.path.join(OUT_DIR, "phase_a1_demo.mp4")
OUT_GIF = os.path.join(OUT_DIR, "phase_a1_demo.gif")

# ── Colors per action ────────────────────────────────────
ACTION_COLORS = {
    "STOP":         (0, 0, 255),      # Red
    "WAIT":         (0, 165, 255),    # Orange
    "SLOW_DOWN":    (0, 215, 255),    # Gold
    "KEEP_CENTER":  (0, 255, 0),      # Green
    "TURN_LEFT":    (255, 200, 0),    # Light blue
    "TURN_RIGHT":   (200, 100, 0),    # Blue
    "BYPASS_LEFT":  (255, 255, 0),    # Cyan
    "BYPASS_RIGHT": (0, 255, 255),    # Yellow
}

ACTION_ICONS = {
    "STOP": "⏹",
    "WAIT": "⏳",
    "SLOW_DOWN": "🐢",
    "KEEP_CENTER": "⬆",
    "BYPASS_LEFT": "⬅",
    "BYPASS_RIGHT": "➡",
}

YOLO_SIZE = (640, 640)
CLASS_COLORS = {}
import random
random.seed(42)

def get_class_color(label):
    if label not in CLASS_COLORS:
        CLASS_COLORS[label] = [random.randint(64, 255) for _ in range(3)]
    return CLASS_COLORS[label]

def draw_frame(result, img_path, out_path):
    """Draw all annotations on one KITTI frame."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARN: cannot read {img_path}")
        return
    img = cv2.resize(img, YOLO_SIZE)
    h, w = img.shape[:2]

    # ── 1. Draw YOLO bboxes ───────────────────────────
    detections = result.get("detections", [])
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        label = d["label"]
        conf = d["confidence"]
        color = get_class_color(label)

        # Bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label: YOLO + InceptionV3
        incept_lbl = d.get("inception_label", "-")
        incept_conf = d.get("inception_confidence", 0)
        txt_yolo = f"{label} {conf:.2f}"
        txt_incept = f"→ {incept_lbl} ({incept_conf:.3f})" if incept_lbl else ""

        # YOLO label on top
        (tw, th), _ = cv2.getTextSize(txt_yolo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw+4, y1), color, -1)
        cv2.putText(img, txt_yolo, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # InceptionV3 label below bbox
        if txt_incept:
            cv2.putText(img, txt_incept, (x1, y2+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── 2. Draw scene info panel (top bar) ─────────────
    scene = result.get("scene_context", {})
    free = scene.get("free_space", {})
    decision = result.get("decision", {})

    panel_h = 90
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:, :] = (40, 40, 40)

    # Frame info
    cv2.putText(panel, f"Frame {result['frame_id']}  |  {result['filename']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Free space bars
    bar_y = 40; bar_h = 20; bar_w = 180
    lanes = [("L", free.get("left", 0), (255, 150, 0)),
             ("C", free.get("center", 0), (0, 220, 0)),
             ("R", free.get("right", 0), (255, 80, 0))]
    bar_x = 10
    for lane_name, val, color in lanes:
        bw = int(bar_w * val)
        pct = int(val * 100)
        # Background
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60, 60, 60), -1)
        # Filled bar
        if bw > 0:
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bw, bar_y+bar_h), color, -1)
        cv2.putText(panel, f"{lane_name}: {pct}%", (bar_x+5, bar_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        bar_x += bar_w + 15

    # Action display
    action = decision.get("action", "N/A")
    conf = decision.get("confidence", 0)
    act_color = ACTION_COLORS.get(action, (255, 255, 255))
    icon = ACTION_ICONS.get(action, "?")
    reason = decision.get("reason", "")

    # Action banner on the right
    act_x = bar_x + 20
    cv2.putText(panel, f"{icon} {action}", (act_x, bar_y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, act_color, 2)
    cv2.putText(panel, f"conf: {conf:.2f}", (act_x, bar_y+36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, reason[:50], (act_x, bar_y+54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Timing
    timing = result.get("timing_ms", {})
    cv2.putText(panel, f"YOLO: {timing.get('yolo', '?')}ms  Total: {timing.get('total', '?')}ms",
                (10, panel_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Obstacle count
    obstacles = scene.get("obstacles", [])
    cv2.putText(panel, f"Obstacles: {len(obstacles)}",
                (bar_x + 20, panel_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ── 3. Draw decision overlay at bottom of image ────
    # Semi-transparent overlay
    overlay = img.copy()
    bar_bottom_h = 50
    cv2.rectangle(overlay, (0, h-bar_bottom_h), (w, h), act_color, -1)
    img = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)

    cv2.putText(img, f"{icon} {action}  ({conf:.1%})",
                (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, act_color, 2)

    # Combine image + panel
    combined = np.vstack([panel, img])

    # ── 5. Save ─────────────────────────────────────────
    cv2.imwrite(str(out_path), combined)

def main():
    if not os.path.exists(SUMMARY):
        print(f"ERROR: summary not found: {SUMMARY}", file=sys.stderr)
        print("Run phase_a1_board.py first.", file=sys.stderr)
        return 1

    with open(SUMMARY) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Processing {len(results)} frames...")

    frames = []
    for r in results:
        fname = r["filename"]
        img_path = os.path.join(KITTI_DIR, fname)
        if not os.path.exists(img_path):
            print(f"  WARN: image not found: {img_path}")
            continue

        out_path = os.path.join(OUT_DIR, f"frame_{r['frame_id']:04d}.jpg")
        draw_frame(r, img_path, out_path)
        frames.append(out_path)
        print(f"  [{r['frame_id']+1}/{len(results)}] {fname} → {r['decision']['action']}")

    # ── Render video with ffmpeg ────────────────────────
    if frames:
        fps = 10
        # Write frame list
        list_path = os.path.join(OUT_DIR, "frames.txt")
        with open(list_path, "w") as f:
            for fp in sorted(frames):
                f.write(f"file '{os.path.abspath(fp)}'\n")
                f.write(f"duration {1/fps}\n")
        # Last frame repeats
        with open(list_path, "a") as f:
            f.write(f"file '{os.path.abspath(frames[-1])}'\n")

        print(f"\nRendering {len(frames)} frames → {OUT_VIDEO} …")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-vsync", "vfr", "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-crf", "23", OUT_VIDEO
        ], check=True, capture_output=True)
        print(f"Video: {OUT_VIDEO}")

        # Also make a GIF
        print(f"Rendering GIF → {OUT_GIF} …")
        subprocess.run([
            "ffmpeg", "-y", "-i", OUT_VIDEO,
            "-vf", "fps=10,scale=960:-1:flags=lanczos",
            "-loop", "0", OUT_GIF
        ], check=True, capture_output=True)
        print(f"GIF: {OUT_GIF}")

    # Stats
    actions = Counter(r["decision"]["action"] for r in results)
    print(f"\nDecision summary:")
    for act, cnt in actions.most_common():
        print(f"  {act}: {cnt}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
