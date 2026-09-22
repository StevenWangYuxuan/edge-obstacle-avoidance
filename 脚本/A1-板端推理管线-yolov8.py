#!/usr/bin/env python3
"""
Phase A1 真机 YOLOv8 版: Mac编排 → 板端 SNPE GPU 推理
KITTI → YOLOv8n DLC → InceptionV3 DLC → scene → decision

与 YOLOv5 版本的关键差异:
  - YOLOv8 输出单个张量 output0: [1, 84, 8400]
  - bbox 是 (x1, y1, x2, y2) 绝对像素坐标，无需转换
  - class score 内置 objectness，无独立 objectness 通道
"""

import json, os, subprocess, sys, time, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

# ── Config ─────────────────────────────────────────────
BOARD_MODELS = "/data/local/tmp/models"
BOARD_WORK = "/data/local/tmp/phase_a1"
SNPE_RUN = "/data/local/tmp/snpe-net-run"
SNPE_LIBS = "/data/local/tmp/qnn_libs"
SNPE_ENV = f"export LD_LIBRARY_PATH={SNPE_LIBS}:$LD_LIBRARY_PATH"

# ═══ YOLOv8 专用配置 ═══
YOLO_DLC = f"{BOARD_MODELS}/yolov8n.dlc"       # YOLOv8n DLC
YOLO_OUTPUT_NAME = "output0"                    # SNPE 输出张量名
YOLO_OUTPUT_RAW = "Result_0/output0.raw"        # snpe-net-run 输出文件
YOLO_OUTPUT_SHAPE = (84, 8400)                  # YOLOv8n 输出: (features, anchors)
INCEPTION_DLC = f"{BOARD_MODELS}/inception_v3.dlc"

YOLO_SIZE = (640, 640)
INCEPTION_SIZE = (299, 299)
MEAN_RGB = (128, 128, 128)      # SNPE 标准预处理: (pixel - mean) / 128
CONF_THRES = 0.25
IOU_THRES = 0.45

LABELS_PATH = os.path.expanduser("~/Documents/端测/项目文档/参考资料/imagenet_slim_labels.txt")

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

OBSTACLE_CLASSES = {
    "car","truck","bus","motorcycle","bicycle","boat","airplane","train",
    "traffic light","stop sign","fire hydrant","parking meter","bench",
    "potted plant","suitcase","backpack","handbag","umbrella","sports ball",
    "dog","cat","horse","sheep","cow","bird",
    "chair","couch","dining table","bed","toilet","tv",
    "refrigerator","microwave","oven","sink",
    "vase","teddy bear","scissors"
}


# ── ADB Helpers ─────────────────────────────────────────
def adb(cmd, timeout=30):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def adb_push(local, remote):
    subprocess.run(["adb", "push", str(local), remote], capture_output=True, check=True)

def adb_pull(remote, local):
    subprocess.run(["adb", "pull", remote, str(local)], capture_output=True, check=True)

def board_run_snpe(dlc, input_list, output_dir, timeout=90):
    """Run snpe-net-run on board. Returns (success, log)."""
    cmd = f"{SNPE_ENV} && {SNPE_RUN} --container {dlc} --input_list {input_list} --output_dir {output_dir}"
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stdout + r.stderr


# ── YOLOv8 Preprocessing (与 YOLOv5 相同) ──────────────
def preprocess_yolo(img_path, raw_path):
    """加载图像, resize 640x640, 归一化 (pixel - 128) / 128, 写 raw."""
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, YOLO_SIZE)
    mean = np.empty((YOLO_SIZE[0], YOLO_SIZE[1], 3), dtype=np.float32)
    for c in range(3): mean[:, :, c] = MEAN_RGB[c]
    raw = (img.astype(np.float32) - mean) / 128.0
    raw.tofile(str(raw_path))
    return img


# ═══════════════════════════════════════════════════════════
# YOLOv8 专用后处理 (与 YOLOv5 完全不同)
# ═══════════════════════════════════════════════════════════

def parse_yolov8(raw_path, conf_thres=0.25, iou_thres=0.45):
    """
    解析 YOLOv8 SNPE 推理输出。
    
    YOLOv8n 输出: output0 形状 [1, 84, 8400]
      - 通道 0-3:  bbox (cx, cy, w, h) — 640x640 上的绝对像素坐标，需转为 (x1,y1,x2,y2)
      - 通道 4-83: 80 个 COCO class 分数 (已含 objectness)
    
    返回: [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
    """
    # 读取 raw float32 → reshape
    raw = np.fromfile(str(raw_path), dtype=np.float32)

    # 支持 [1,84,8400] 或 [84,8400]
    if raw.size == 84 * 8400:
        data = raw.reshape(84, 8400).T      # → [8400, 84]
    elif raw.size == 1 * 84 * 8400:
        data = raw.reshape(1, 84, 8400)
        data = data[0].T                     # → [8400, 84]
    else:
        print(f"  WARN: unexpected raw size {raw.size}, expected {84*8400}")
        return []

    if data.size == 0:
        return []

    # 拆分 bbox: 实际输出为 (cx, cy, w, h)，转为 (x1, y1, x2, y2)
    cx, cy, bw, bh = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    bboxes_raw = np.stack([
        cx - bw / 2,   # x1
        cy - bh / 2,   # y1
        cx + bw / 2,   # x2
        cy + bh / 2    # y2
    ], axis=1)          # [8400, 4]  (x1, y1, x2, y2)
    class_scores = data[:, 4:84]      # [8400, 80]

    # 每行取 max class score + class id
    max_scores = np.max(class_scores, axis=1)      # [8400]
    best_class_ids = np.argmax(class_scores, axis=1)  # [8400]

    # 按置信度过滤
    mask = max_scores > conf_thres
    if not np.any(mask):
        return []

    bboxes = bboxes_raw[mask]           # [N, 4]
    scores = max_scores[mask]           # [N]
    class_ids = best_class_ids[mask]    # [N]

    # ── NMS per class ──
    keep_indices = _nms_per_class(bboxes, scores, class_ids, iou_thres)
    if len(keep_indices) == 0:
        return []

    bboxes = bboxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]

    # 构建检测结果
    dets = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i].astype(float)
        cls_id = int(class_ids[i])
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"
        dets.append({
            "label": label,
            "confidence": round(float(scores[i]), 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return dets


def _nms_per_class(boxes, scores, class_ids, iou_thres):
    """按类别分别执行 NMS，返回保留下来的全局索引列表。"""
    keep = []
    for cls_id in np.unique(class_ids):
        m = class_ids == cls_id
        idx = np.where(m)[0]
        order = scores[idx].argsort()[::-1]     # 同一类内按分数降序
        idx = idx[order]

        while len(idx) > 0:
            keep.append(idx[0])
            if len(idx) == 1:
                break
            # IoU 计算: (x1, y1, x2, y2)
            box = boxes[idx[0]]
            rest = boxes[idx[1:]]
            xx1 = np.maximum(box[0], rest[:, 0])
            yy1 = np.maximum(box[1], rest[:, 1])
            xx2 = np.minimum(box[2], rest[:, 2])
            yy2 = np.minimum(box[3], rest[:, 3])
            iw = np.maximum(0, xx2 - xx1)
            ih = np.maximum(0, yy2 - yy1)
            inter = iw * ih
            area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            iou = inter / np.maximum(area_box + area_rest - inter, 1e-6)
            idx = idx[1:][iou <= iou_thres]

    return keep


# ── InceptionV3 (与 YOLOv5 版相同) ─────────────────────
def preprocess_inception_crop(img, bbox, raw_path):
    x1, y1, x2, y2 = [max(0, int(round(v))) for v in bbox]
    x2 = min(YOLO_SIZE[0], x2)
    y2 = min(YOLO_SIZE[1], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.resize(img[y1:y2, x1:x2], INCEPTION_SIZE)
    mean = np.empty((INCEPTION_SIZE[0], INCEPTION_SIZE[1], 3), dtype=np.float32)
    for c in range(3): mean[:, :, c] = MEAN_RGB[c]
    raw = (crop.astype(np.float32) - mean) / 128.0
    raw.tofile(str(raw_path))
    return raw_path


# ── Scene → Decision (与 YOLOv5 版相同) ────────────────
def detections_to_scene(fid, detections):
    obstacles = []
    for d in detections:
        lbl = d.get("inception_label") or d["label"]
        if lbl not in OBSTACLE_CLASSES and d["label"] not in OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = d["bbox"]
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        cx = (x1 + x2) / 2 / YOLO_SIZE[0]
        cy = (y1 + y2) / 2 / YOLO_SIZE[1]
        nw = bw / YOLO_SIZE[0]
        nh = bh / YOLO_SIZE[1]
        lane = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        dist = round(min(80.0, 0.43 / max(nh, 0.001)), 2)
        obstacles.append({
            "type": lbl, "yolo_label": d["label"], "yolo_conf": d["confidence"],
            "inception_label": d.get("inception_label"),
            "inception_conf": d.get("inception_confidence", 0),
            "x": round(cx, 3), "y": round(cy, 3), "w": round(nw, 3), "h": round(nh, 3),
            "distance_est": dist, "lane_hint": lane, "confidence": d["confidence"]
        })
    lanes = {"left": 0.0, "center": 0.0, "right": 0.0}
    for o in obstacles:
        lanes[o["lane_hint"]] += o["w"] * (0.65 if o["distance_est"] < 2.0 else 0.35)
    fs = {n: round(max(0.0, min(1.0, 1.0 - min(v, 0.95))), 2) for n, v in lanes.items()}
    return {
        "frame_id": fid, "obstacles": obstacles, "free_space": fs,
        "ego_state": {"speed_mps": 1.0, "speed_level": "slow", "heading": "forward"}
    }


def decide(scene):
    free = scene["free_space"]
    ego = scene["ego_state"]
    obs = scene.get("obstacles", [])
    nearest = min((o["distance_est"] for o in obs), default=999.0)
    co = [o for o in obs if o["lane_hint"] == "center"]
    nc = min((o["distance_est"] for o in co), default=999.0)
    cb = free["center"] < 0.25 or nc < 1.8
    lb = free["left"] - free["right"] > 0.12
    rb = free["right"] - free["left"] > 0.12

    if nearest < 0.8:
        return {"action": "STOP", "confidence": 0.98, "reason": f"emergency:{nearest:.2f}m", "source": "rule_engine"}
    if cb and free["left"] < 0.25 and free["right"] < 0.25:
        if ego["speed_level"] in {"medium", "fast"} or nc < 1.2:
            return {"action": "STOP", "confidence": 0.93, "reason": "center blocked, sides unsafe", "source": "rule_engine"}
        return {"action": "WAIT", "confidence": 0.81, "reason": "center blocked, sides narrow", "source": "rule_engine"}
    if nc < 2.0 and cb:
        if lb and free["left"] > 0.40:
            c = round(min(0.55 + min(free["left"], 0.9) * 0.25 + min(max(nc, 0), 3) * 0.06, 0.95), 2)
            return {"action": "BYPASS_LEFT", "confidence": c, "reason": "left safer", "source": "rule_engine"}
        if rb and free["right"] > 0.40:
            c = round(min(0.55 + min(free["right"], 0.9) * 0.25 + min(max(nc, 0), 3) * 0.06, 0.95), 2)
            return {"action": "BYPASS_RIGHT", "confidence": c, "reason": "right safer", "source": "rule_engine"}
        return {"action": "SLOW_DOWN", "confidence": 0.72, "reason": "center obstacle, slowing", "source": "rule_engine"}
    if free["center"] >= 0.45 and nearest > 2.0:
        return {"action": "KEEP_CENTER", "confidence": 0.88, "reason": "center clear", "source": "rule_engine"}
    if free["left"] > 0.55 and lb:
        return {"action": "BYPASS_LEFT", "confidence": 0.70, "reason": "left wider", "source": "rule_engine"}
    if free["right"] > 0.55 and rb:
        return {"action": "BYPASS_RIGHT", "confidence": 0.70, "reason": "right wider", "source": "rule_engine"}
    return {"action": "SLOW_DOWN", "confidence": 0.60, "reason": "ambiguous fallback", "source": "rule_engine"}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    imagenet_labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            imagenet_labels = [l.strip() for l in f if l.strip()]
    print(f"Imagenet labels loaded: {len(imagenet_labels)}")

    KITTI = os.path.expanduser("~/Documents/端测/测试数据/行车记录仪帧")
    OUT = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU")
    os.makedirs(OUT, exist_ok=True)

    images = sorted(Path(KITTI).glob("*.png"))
    print(f"Phase A1 YOLOv8n: {len(images)} frames → YOLOv8n → InceptV3 → decision\n")

    # ── ADB check ──
    if "OK" not in adb("echo OK"):
        print("ERROR: adb not responding", file=sys.stderr)
        return 1

    # Verify DLC present on board
    check = adb(f"ls -lh {YOLO_DLC} 2>/dev/null || echo MISSING")
    if "MISSING" in check:
        print(f"ERROR: {YOLO_DLC} not found on board!", file=sys.stderr)
        print("Push it first: adb push yolov8n.dlc /data/local/tmp/models/", file=sys.stderr)
        return 1
    print(f"Board DLC: {check}\n")

    adb(f"rm -rf {BOARD_WORK} && mkdir -p {BOARD_WORK}")

    all_results = []
    t0_total = time.time()

    for idx, img_path in enumerate(images):
        tag = img_path.stem
        print(f"[{idx + 1}/{len(images)}] Frame {tag} …", end="", flush=True)
        t0 = time.time()

        try:
            with tempfile.TemporaryDirectory() as tmp:
                # ═══ 1. YOLOv8 预处理 ═══
                raw_local = Path(tmp) / "yolo.raw"
                yolo_img = preprocess_yolo(img_path, raw_local)
                board_raw = f"{BOARD_WORK}/f{tag}.raw"
                adb_push(raw_local, board_raw)

                # ═══ 2. YOLOv8 推理 ═══
                list_local = Path(tmp) / "list.txt"
                list_local.write_text(board_raw + "\n")
                board_list = f"{BOARD_WORK}/f{tag}_list.txt"
                adb_push(list_local, board_list)

                yolo_out = f"{BOARD_WORK}/yolo_{tag}"
                ok, log = board_run_snpe(YOLO_DLC, board_list, yolo_out)
                t_yolo = round((time.time() - t0) * 1000)

                if not ok:
                    print(f" YOLO FAILED: {log[-200:]}", flush=True)
                    continue

                # ═══ Pull & parse YOLOv8 ═══
                yolo_remote_out = f"{yolo_out}/{YOLO_OUTPUT_RAW}"
                yolo_local_out = Path(tmp) / "yolo_out.raw"
                adb_pull(yolo_remote_out, yolo_local_out)

                if yolo_local_out.exists():
                    dets = parse_yolov8(yolo_local_out, CONF_THRES, IOU_THRES)
                else:
                    print(f"\n  WARN: YOLO output not found at {yolo_remote_out}", flush=True)
                    dets = []

                ob_dets = [d for d in dets if d["label"] in OBSTACLE_CLASSES]
                print(f" YOLO:{len(dets)}(ob:{len(ob_dets)})@{t_yolo}ms", flush=True)

                # ═══ 3. InceptionV3 per bbox ═══
                for i, d in enumerate(dets):
                    crop_local = Path(tmp) / f"crop_{i}.raw"
                    r = preprocess_inception_crop(yolo_img, d["bbox"], crop_local)
                    if r is None:
                        d.update(inception_label=None, inception_confidence=0.0)
                        continue

                    board_crop = f"{BOARD_WORK}/f{tag}_c{i}.raw"
                    adb_push(crop_local, board_crop)
                    cl_local = Path(tmp) / f"cl_{i}.txt"
                    cl_local.write_text(board_crop + "\n")
                    board_cl = f"{BOARD_WORK}/f{tag}_cl_{i}.txt"
                    adb_push(cl_local, board_cl)

                    incept_out = f"{BOARD_WORK}/in_{tag}_{i}"
                    board_run_snpe(INCEPTION_DLC, board_cl, incept_out)

                    adb_pull(
                        f"{incept_out}/Result_0/InceptionV3/Predictions/Reshape_1:0.raw",
                        Path(tmp) / f"in_{i}.raw"
                    )
                    ip = Path(tmp) / f"in_{i}.raw"
                    if ip.exists():
                        probs = np.fromfile(str(ip), dtype=np.float32)
                        if len(probs) > 0:
                            ti = int(np.argmax(probs))
                            d["inception_confidence"] = round(float(probs[ti]), 4)
                            d["inception_label"] = imagenet_labels[ti] if ti < len(imagenet_labels) else f"c{ti}"
                        else:
                            d.update(inception_label=None, inception_confidence=0.0)
                    else:
                        d.update(inception_label=None, inception_confidence=0.0)

                for d in ob_dets:
                    il = d.get("inception_label", "-")
                    ic = d.get("inception_confidence", 0)
                    print(f"  {d['label']}→{il}({ic:.3f})", flush=True)

                # ═══ 4. Scene + Decision ═══
                scene = detections_to_scene(idx, dets)
                dec = decide(scene)
                t_tot = round((time.time() - t0) * 1000)
                print(f"  → {dec['action']} conf={dec['confidence']} [{t_tot}ms]\n", flush=True)

                all_results.append({
                    "frame_id": idx,
                    "filename": img_path.name,
                    "detections": dets,
                    "scene_context": scene,
                    "decision": dec,
                    "timing_ms": {"total": t_tot, "yolo": t_yolo},
                })

                # Cleanup board
                adb(f"rm -rf {BOARD_WORK}/f{tag}* {BOARD_WORK}/yolo_{tag} {BOARD_WORK}/in_{tag}_*", timeout=5)

        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            # Save partial progress
            partial_path = Path(OUT) / "phase_a1_board_summary_partial.json"
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  Partial results saved ({len(all_results)} frames), continuing...\n", flush=True)
            continue

    # ── Summary ──────────────────────────────────────────
    total_t = time.time() - t0_total
    print(f"{'=' * 60}\nDone: {len(all_results)} frames in {total_t:.1f}s")

    summary_path = Path(OUT) / "phase_a1_board_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")

    acts = Counter(r["decision"]["action"] for r in all_results)
    print("Decisions:")
    for a, c in acts.most_common():
        print(f"  {a}: {c}")

    incept_n = sum(1 for r in all_results for d in r["detections"] if d.get("inception_label"))
    print(f"InceptionV3 enriched: {incept_n} detections")

    yt = [r["timing_ms"]["yolo"] for r in all_results]
    tt = [r["timing_ms"]["total"] for r in all_results]
    if yt:
        print(f"YOLOv8 avg: {sum(yt) // len(yt)}ms, Total avg: {sum(tt) // len(tt)}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
