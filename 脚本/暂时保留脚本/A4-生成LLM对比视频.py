#!/usr/bin/env python3
"""
Phase A4 visualization: side-by-side LLM vs Rule engine comparison
+ YOLO bboxes + InceptionV3 labels + free-space meters
"""

import json, os, sys, subprocess
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

SUMMARY_A4 = os.path.expanduser("~/Documents/端测/输出/行车记录仪-LLM决策结果-v4/phase_a4_results_v4.json")
SUMMARY_A1 = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU/phase_a1_board_summary.json")
KITTI_DIR = os.path.expanduser("~/Documents/端测/测试数据/行车记录仪帧")
OUT_DIR = os.path.expanduser("~/Documents/端测/输出/LLM对比可视化-行车记录仪-v4")

ACTION_COLORS = {
    "STOP": (0, 0, 255), "WAIT": (0, 165, 255),
    "SLOW_DOWN": (0, 215, 255), "KEEP_CENTER": (0, 255, 0),
    "BYPASS_LEFT": (255, 255, 0), "BYPASS_RIGHT": (0, 255, 255),
}

YOLO_SIZE = (640, 640)
import random; random.seed(42)
CLASS_COLORS = {}

def cls_color(l):
    if l not in CLASS_COLORS: CLASS_COLORS[l] = [random.randint(64,255) for _ in range(3)]
    return CLASS_COLORS[l]

def draw_frame(a1_result, a4_result, img_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None: return
    img = cv2.resize(img, YOLO_SIZE)
    h, w = img.shape[:2]
    dets = a1_result.get("detections", [])
    scene = a1_result.get("scene_context", {})
    free = scene.get("free_space", {})
    dec_rule = a1_result.get("decision", {})
    dec_llm = a4_result.get("decision_llm", {}) if a4_result else {}

    # Bboxes
    for d in dets:
        x1,y1,x2,y2 = [max(0,int(round(v))) for v in d["bbox"]]
        x2=min(w,x2); y2=min(h,y2)
        c = cls_color(d["label"])
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
        txt = f'{d["label"]} {d["confidence"]:.2f}'
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1,y1-th-3), (x1+tw+2,y1), c, -1)
        cv2.putText(img, txt, (x1+1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
        il = d.get("inception_label","")
        if il:
            cv2.putText(img, f'→{il}({d.get("inception_confidence",0):.2f})', (x1,y2+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180),1)

    # Panel
    ph = 120; panel = np.full((ph,w,3), (35,35,35), dtype=np.uint8)
    cv2.putText(panel, f'Frame {a1_result["frame_id"]} | {a1_result["filename"]}',
                (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)

    # Free-space bars
    bar_x=10; bar_y=30; bar_w=170; bar_h=16
    for lane,val,color in [("L",free.get("left",0),(255,150,0)),
                            ("C",free.get("center",0),(0,220,0)),
                            ("R",free.get("right",0),(255,80,0))]:
        bw=int(bar_w*val)
        cv2.rectangle(panel,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),(55,55,55),-1)
        if bw>0: cv2.rectangle(panel,(bar_x,bar_y),(bar_x+bw,bar_y+bar_h),color,-1)
        cv2.putText(panel, f'{lane}:{int(val*100)}%',(bar_x+3,bar_y+12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        bar_x+=bar_w+12

    # Side-by-side: Rule Engine (left) vs LLM (right)
    cmp_y = bar_y+bar_h+15
    # Left column: Rule Engine
    cv2.rectangle(panel, (10,cmp_y), (w//2-8,ph-8), (60,60,60), 2)
    cv2.putText(panel, '⚙ Rule Engine', (16,cmp_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    ra = dec_rule.get("action","?"); rc = dec_rule.get("confidence",0); rr = dec_rule.get("reason","")
    ra_c = ACTION_COLORS.get(ra,(200,200,200))
    cv2.putText(panel, ra, (16,cmp_y+42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ra_c, 2)
    cv2.putText(panel, f'conf:{rc:.2f}', (16,cmp_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1)

    # Right column: LLM
    cv2.rectangle(panel, (w//2+2,cmp_y), (w-10,ph-8), (60,55,65), 2)
    cv2.putText(panel, '🧠 TinyLlama AI', (w//2+8,cmp_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
    if dec_llm:
        la = dec_llm.get("action","?"); lc = dec_llm.get("confidence",0); lr = dec_llm.get("reason","")
        la_c = ACTION_COLORS.get(la,(200,200,200))
        match = "✅" if la == ra else "⚠️"
        cv2.putText(panel, f'{la} {match}', (w//2+8,cmp_y+42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, la_c, 2)
        cv2.putText(panel, f'conf:{lc:.2f}', (w//2+8,cmp_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1)
    else:
        cv2.putText(panel, 'N/A', (w//2+8,cmp_y+42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120),1)

    # Obstacle count + timing
    obs = scene.get("obstacles",[])
    timing = a1_result.get("timing_ms",{})
    cv2.putText(panel, f'Obstacles: {len(obs)} | YOLO: {timing.get("yolo","?")}ms',
                (10,ph-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120),1)

    # Bottom action banner
    combined = np.vstack([panel, img])
    la = dec_llm.get("action","?") if dec_llm else "?"
    if dec_llm and la == ra: action = f'{ra} (agreed)'
    else: action = f'{ra} / LLM:{la}'
    c2 = combined.copy()
    cv2.rectangle(c2, (0,h+ph-36), (w,h+ph), ACTION_COLORS.get(ra,(0,200,0)), -1)
    final = cv2.addWeighted(combined, 0.82, c2, 0.18, 0)
    cv2.putText(final, action, (12,h+ph-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imwrite(str(out_path), final)

def main():
    if not os.path.exists(SUMMARY_A1):
        print("Missing A1 summary", file=sys.stderr); return 1
    with open(SUMMARY_A1) as f: a1_data = json.load(f)
    a4_data = {}
    if os.path.exists(SUMMARY_A4):
        with open(SUMMARY_A4) as f:
            a4_list = json.load(f)
        for r in a4_list: a4_data[r["frame_idx"]] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Rendering {len(a1_data)} frames with LLM vs Rule comparison...")

    for r in a1_data:
        fid = r["frame_id"]
        fname = r["filename"]
        img_path = os.path.join(KITTI_DIR, fname)
        a4 = a4_data.get(fid)
        out = os.path.join(OUT_DIR, f"frame_{fid:04d}.jpg")
        draw_frame(r, a4, img_path, out)
        la = a4["decision_llm"]["action"] if a4 else "N/A"
        ra = r["decision"]["action"]
        m = "✅" if la == ra else "⚠️"
        print(f"  [{fid+1}/{len(a1_data)}] {fname}  Rule:{ra} vs LLM:{la} {m}")

    # Video
    list_path = os.path.join(OUT_DIR, "frames.txt")
    frames_sorted = sorted(Path(OUT_DIR).glob("frame_*.jpg"))
    with open(list_path, "w") as f:
        for fp in frames_sorted: f.write(f"file '{os.path.abspath(fp)}'\nduration 0.1\n")
        f.write(f"file '{os.path.abspath(frames_sorted[-1])}'\n")

    video_path = os.path.join(OUT_DIR, "phase_a4_comparison.mp4")
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",list_path,
                     "-vsync","vfr","-pix_fmt","yuv420p","-c:v","libx264","-crf","23",video_path],
                   capture_output=True,check=True)
    print(f"Video: {video_path}")

    # Stats
    matches = sum(1 for r in a1_data if a4_data.get(r["frame_id"],{}).get("decision_llm",{}).get("action") == r["decision"]["action"])
    total = sum(1 for r in a1_data if a4_data.get(r["frame_id"]))
    print(f"\nAgreement: {matches}/{total} ({matches/total*100:.0f}%)" if total else "")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
