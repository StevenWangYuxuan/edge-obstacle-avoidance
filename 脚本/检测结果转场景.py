#!/usr/bin/env python3
"""Convert YOLO/classification detections into a SceneContext JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_detections(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    required = {"frame_id", "image_width", "image_height", "detections"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing required keys: {sorted(missing)}")
    return payload


def lane_hint(x_center: float) -> str:
    if x_center < 1.0 / 3.0:
        return "left"
    if x_center > 2.0 / 3.0:
        return "right"
    return "center"


def distance_est(height_norm: float, width_norm: float) -> float:
    """Perspective-based distance: d = k / h.
    Calibrated for 640x640 dashcam (~120deg HFOV), car ~1.5m tall.
    k = focal_px * real_h / img_h ≈ 185 * 1.5 / 640 ≈ 0.43
    """
    return round(min(80.0, 0.43 / max(height_norm, 0.001)), 2)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_free_space(obstacles: list[dict]) -> dict:
    lanes = {"left": 0.0, "center": 0.0, "right": 0.0}
    for item in obstacles:
        penalty = item["w"] * (0.65 if item["distance_est"] < 2.0 else 0.35)
        lanes[item["lane_hint"]] += penalty

    return {
        name: round(clamp(1.0 - min(occ, 0.95)), 2)
        for name, occ in lanes.items()
    }


def convert(payload: dict, speed_mps: float, heading: str) -> dict:
    width = payload["image_width"]
    height = payload["image_height"]
    obstacles = []

    for item in payload["detections"]:
        x1, y1, x2, y2 = item["bbox"]
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)
        center_x = (x1 + x2) / 2.0 / width
        center_y = (y1 + y2) / 2.0 / height
        norm_w = bbox_w / width
        norm_h = bbox_h / height

        obstacles.append(
            {
                "type": item["label"],
                "x": round(center_x, 3),
                "y": round(center_y, 3),
                "w": round(norm_w, 3),
                "h": round(norm_h, 3),
                "distance_est": distance_est(norm_h, norm_w),
                "lane_hint": lane_hint(center_x),
                "confidence": round(item["confidence"], 3),
            }
        )

    free_space = compute_free_space(obstacles)
    speed_level = "stop" if speed_mps == 0 else "slow" if speed_mps < 1.5 else "medium" if speed_mps < 4.0 else "fast"

    return {
        "frame_id": payload["frame_id"],
        "obstacles": obstacles,
        "free_space": free_space,
        "ego_state": {
            "speed_mps": round(speed_mps, 2),
            "speed_level": speed_level,
            "heading": heading,
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) not in {2, 4}:
        print(f"usage: {argv[0]} <detections.json> [speed_mps heading]", file=sys.stderr)
        return 2

    speed_mps = 1.0
    heading = "forward"
    if len(argv) == 4:
        speed_mps = float(argv[2])
        heading = argv[3]

    scene = convert(load_detections(Path(argv[1])), speed_mps=speed_mps, heading=heading)
    print(json.dumps(scene, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
