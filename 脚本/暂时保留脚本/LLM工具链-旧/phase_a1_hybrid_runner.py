#!/usr/bin/env python3
"""
Hybrid Phase A1 runner.

Local machine:
- reads KITTI frames
- preprocesses YOLO/Inception inputs with PIL + numpy
- postprocesses outputs
- builds scene_context and decision

Remote VM:
- runs SNPE inference for YOLOv5s and InceptionV3 via ssh/scp
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from realtime_decision import decide


LOCAL_ROOT = Path("/Users/steven/Documents/端测")
KITTI_DIR = LOCAL_ROOT / "2011_09_26/2011_09_26_drive_0048_extract/image_02/data"
OUTPUT_DIR = LOCAL_ROOT / "phase_a1_hybrid_output"

REMOTE_HOST = "steven@192.168.64.3"
REMOTE_HOME = "/home/steven"
REMOTE_BASE = f"{REMOTE_HOME}/phase_a1_bridge"
REMOTE_SNPE_ROOT = f"{REMOTE_HOME}/qairt/2.22.6.240515"
REMOTE_SNPE_BIN = f"{REMOTE_SNPE_ROOT}/bin/x86_64-linux-clang/snpe-net-run"
REMOTE_SNPE_LIB = f"{REMOTE_SNPE_ROOT}/lib/x86_64-linux-clang"
REMOTE_YOLO_DLC = f"{REMOTE_HOME}/day4-yolo/01_models/yolov5s_quantized.dlc"
REMOTE_INCEPTION_DLC = (
    f"{REMOTE_SNPE_ROOT}/examples/Models/InceptionV3/tensorflow/inception_v3.dlc"
)
REMOTE_INCEPTION_LABELS = (
    f"{REMOTE_SNPE_ROOT}/examples/Models/InceptionV3/tensorflow/imagenet_slim_labels.txt"
)

YOLO_SIZE = (640, 640)
INCEPTION_SIZE = (299, 299)
MEAN_RGB = np.array([128.0, 128.0, 128.0], dtype=np.float32)
CONF_THRES = 0.25
IOU_THRES = 0.45

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "dog", "cat", "horse", "sheep", "cow", "bird",
    "chair", "couch", "potted plant", "dining table", "tv", "sink", "refrigerator",
}


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def remote_bash(command: str) -> subprocess.CompletedProcess:
    return run(["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(command)}"])


def scp_to(local_path: Path, remote_path: str) -> subprocess.CompletedProcess:
    return run(["scp", str(local_path), f"{REMOTE_HOST}:{remote_path}"])


def scp_from(remote_path: str, local_path: Path) -> subprocess.CompletedProcess:
    return run(["scp", f"{REMOTE_HOST}:{remote_path}", str(local_path)])


def load_imagenet_labels() -> list[str]:
    target = OUTPUT_DIR / "imagenet_slim_labels.txt"
    if not target.exists():
        result = scp_from(REMOTE_INCEPTION_LABELS, target)
        if result.returncode != 0:
            raise RuntimeError(f"failed to fetch labels: {result.stderr.strip()}")
    return [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image_path: Path, size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(image_path).convert("RGB").resize(size)
    array = np.asarray(image, dtype=np.float32)
    raw = (array - MEAN_RGB) / 128.0
    return image, raw


def save_raw(array: np.ndarray, path: Path) -> None:
    array.astype(np.float32).tofile(path)


def filter_boxes(org_box: np.ndarray) -> np.ndarray:
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > CONF_THRES
    box = org_box[conf]
    if len(box) == 0:
        return np.array([])

    cls_scores = box[..., 5:]
    best_cls = np.argmax(cls_scores, axis=1)
    best_score = np.max(cls_scores, axis=1)
    box[:, 4] = best_score
    box[:, 5] = best_cls
    box = box[:, :6]

    output = []
    for cls_id in np.unique(best_cls).astype(int):
        cls_boxes = box[box[:, 5] == cls_id]
        y = np.copy(cls_boxes)
        y[:, 0] = cls_boxes[:, 0] - cls_boxes[:, 2] / 2
        y[:, 1] = cls_boxes[:, 1] - cls_boxes[:, 3] / 2
        y[:, 2] = cls_boxes[:, 0] + cls_boxes[:, 2] / 2
        y[:, 3] = cls_boxes[:, 1] + cls_boxes[:, 3] / 2

        x1, y1, x2, y2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        scores = y[:, 4]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)
            idx = np.where(ious <= IOU_THRES)[0]
            order = order[idx + 1]
        output.append(y[keep])
    return np.concatenate(output, axis=0) if output else np.array([])


def parse_yolo_output(raw_path: Path) -> list[dict]:
    data = np.fromfile(raw_path, dtype=np.float32).reshape(1, 25200, 85)
    boxes = filter_boxes(data)
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4].astype(float)
        cls_id = int(box[5])
        detections.append(
            {
                "label": CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}",
                "confidence": round(float(box[4]), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return detections


def lane_hint(center_x: float) -> str:
    if center_x < 1.0 / 3.0:
        return "left"
    if center_x > 2.0 / 3.0:
        return "right"
    return "center"


def distance_est(height_norm: float, width_norm: float) -> float:
    area = max(height_norm * width_norm, 1e-4)
    return round(min(8.0, 1.6 / area), 2)


def detections_to_scene(frame_id: int, detections: list[dict]) -> dict:
    obstacles = []
    for item in detections:
        label = item.get("inception_label") or item["label"]
        if label not in OBSTACLE_CLASSES and item["label"] not in OBSTACLE_CLASSES:
            continue

        x1, y1, x2, y2 = item["bbox"]
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)
        center_x = (x1 + x2) / 2.0 / YOLO_SIZE[0]
        center_y = (y1 + y2) / 2.0 / YOLO_SIZE[1]
        norm_w = bbox_w / YOLO_SIZE[0]
        norm_h = bbox_h / YOLO_SIZE[1]
        obstacles.append(
            {
                "type": label,
                "yolo_label": item["label"],
                "yolo_conf": item["confidence"],
                "inception_label": item.get("inception_label"),
                "inception_conf": item.get("inception_confidence", 0.0),
                "x": round(center_x, 3),
                "y": round(center_y, 3),
                "w": round(norm_w, 3),
                "h": round(norm_h, 3),
                "distance_est": distance_est(norm_h, norm_w),
                "lane_hint": lane_hint(center_x),
                "confidence": item["confidence"],
            }
        )

    lanes = {"left": 0.0, "center": 0.0, "right": 0.0}
    for item in obstacles:
        penalty = item["w"] * (0.65 if item["distance_est"] < 2.0 else 0.35)
        lanes[item["lane_hint"]] += penalty

    free_space = {
        name: round(max(0.0, min(1.0, 1.0 - min(occ, 0.95))), 2)
        for name, occ in lanes.items()
    }
    return {
        "frame_id": frame_id,
        "obstacles": obstacles,
        "free_space": free_space,
        "ego_state": {"speed_mps": 1.0, "speed_level": "slow", "heading": "forward"},
    }


def scene_for_rule_engine(scene: dict) -> dict:
    slim_obstacles = []
    for item in scene["obstacles"]:
        slim_obstacles.append(
            {
                "type": item["type"],
                "x": item["x"],
                "y": item["y"],
                "w": item["w"],
                "h": item["h"],
                "distance_est": item["distance_est"],
                "lane_hint": item["lane_hint"],
                "confidence": item["confidence"],
            }
        )
    return {
        "frame_id": scene["frame_id"],
        "obstacles": slim_obstacles,
        "free_space": scene["free_space"],
        "ego_state": scene["ego_state"],
    }


def find_remote_output(remote_dir: str, filename: str) -> str:
    result = remote_bash(f"find {remote_dir} -name {filename} | head -1")
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"remote output not found: {filename}\n{result.stderr.strip()}")
    return result.stdout.strip()


def run_remote_snpe(container: str, raw_local: Path, remote_frame_dir: str, tag: str) -> Path:
    remote_raw = f"{remote_frame_dir}/{tag}.raw"
    remote_list = f"{remote_frame_dir}/{tag}_input_list.txt"
    scp = scp_to(raw_local, remote_raw)
    if scp.returncode != 0:
        raise RuntimeError(f"scp to remote failed: {scp.stderr.strip()}")

    setup = remote_bash(
        f"mkdir -p {remote_frame_dir}/out && printf '%s\\n' {shlex.quote(remote_raw)} > {remote_list}"
    )
    if setup.returncode != 0:
        raise RuntimeError(f"remote setup failed: {setup.stderr.strip()}")

    infer = remote_bash(
        f"export LD_LIBRARY_PATH={REMOTE_SNPE_LIB}:$LD_LIBRARY_PATH && "
        f"rm -rf {remote_frame_dir}/out/* && "
        f"{REMOTE_SNPE_BIN} --container {container} --input_list {remote_list} --output_dir {remote_frame_dir}/out"
    )
    if infer.returncode != 0:
        raise RuntimeError(f"remote snpe failed: {infer.stderr.strip() or infer.stdout.strip()}")

    filename = "output0.raw" if "yolo" in tag else "Reshape_1:0.raw"
    remote_output = find_remote_output(f"{remote_frame_dir}/out", filename)
    local_output = OUTPUT_DIR / f"{tag}_{Path(filename).name.replace(':', '_')}"
    fetch = scp_from(remote_output, local_output)
    if fetch.returncode != 0:
        raise RuntimeError(f"scp from remote failed: {fetch.stderr.strip()}")
    return local_output


def enrich_with_inception(
    image: Image.Image, detections: list[dict], labels: list[str], remote_frame_dir: str, frame_tag: str
) -> list[dict]:
    enriched = []
    for idx, det in enumerate(detections):
        bbox = [int(round(v)) for v in det["bbox"]]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(YOLO_SIZE[0], x1))
        y1 = max(0, min(YOLO_SIZE[1], y1))
        x2 = max(0, min(YOLO_SIZE[0], x2))
        y2 = max(0, min(YOLO_SIZE[1], y2))
        if x2 <= x1 or y2 <= y1:
            det["inception_label"] = None
            det["inception_confidence"] = 0.0
            enriched.append(det)
            continue

        crop = image.crop((x1, y1, x2, y2)).resize(INCEPTION_SIZE)
        crop_array = (np.asarray(crop, dtype=np.float32) - MEAN_RGB) / 128.0
        crop_raw = OUTPUT_DIR / f"{frame_tag}_crop_{idx}.raw"
        save_raw(crop_array, crop_raw)

        try:
            output = run_remote_snpe(
                REMOTE_INCEPTION_DLC,
                crop_raw,
                f"{remote_frame_dir}/inception_{idx}",
                f"{frame_tag}_inception_{idx}",
            )
            probs = np.fromfile(output, dtype=np.float32)
            if probs.size:
                top_idx = int(np.argmax(probs))
                det["inception_label"] = labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}"
                det["inception_confidence"] = round(float(probs[top_idx]), 4)
            else:
                det["inception_label"] = None
                det["inception_confidence"] = 0.0
        except Exception:
            det["inception_label"] = None
            det["inception_confidence"] = 0.0

        enriched.append(det)
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3, help="Number of KITTI frames to test")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = sorted(KITTI_DIR.glob("*.png"))
    if not image_files:
        print(f"no images found in {KITTI_DIR}", file=sys.stderr)
        return 1

    probe = remote_bash(
        f"mkdir -p {REMOTE_BASE} && "
        f"test -x {REMOTE_SNPE_BIN} && "
        f"test -d {REMOTE_SNPE_LIB} && echo ok"
    )
    if probe.returncode != 0 or "ok" not in probe.stdout:
        print(f"remote VM not ready:\n{probe.stderr}", file=sys.stderr)
        return 1

    labels = load_imagenet_labels()
    selected = image_files[: args.frames]
    all_results = []
    for frame_id, image_path in enumerate(selected):
        frame_tag = image_path.stem
        remote_frame_dir = f"{REMOTE_BASE}/{frame_tag}"
        remote_bash(f"mkdir -p {remote_frame_dir}")

        image, yolo_raw = preprocess_image(image_path, YOLO_SIZE)
        raw_path = OUTPUT_DIR / f"{frame_tag}_yolo.raw"
        save_raw(yolo_raw, raw_path)

        yolo_output = run_remote_snpe(REMOTE_YOLO_DLC, raw_path, remote_frame_dir, f"{frame_tag}_yolo")
        detections = parse_yolo_output(yolo_output)
        detections = enrich_with_inception(image, detections, labels, remote_frame_dir, frame_tag)
        scene = detections_to_scene(frame_id, detections)
        decision = decide(scene_for_rule_engine(scene))

        result = {
            "frame_id": frame_id,
            "filename": image_path.name,
            "detections": detections,
            "scene_context": scene,
            "decision": decision,
        }
        out_path = OUTPUT_DIR / f"{frame_tag}_result.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        all_results.append(result)

        obstacle_count = len(scene["obstacles"])
        print(f"{image_path.name}: obstacles={obstacle_count} action={decision['action']} reason={decision['reason']}")

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
