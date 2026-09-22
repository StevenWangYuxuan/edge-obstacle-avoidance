#!/usr/bin/env python3
"""Deterministic real-time fallback controller for obstacle avoidance."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ACTIONS = {
    "stop": "STOP",
    "wait": "WAIT",
    "slow": "SLOW_DOWN",
    "keep": "KEEP_CENTER",
    "left": "BYPASS_LEFT",
    "right": "BYPASS_RIGHT",
}


@dataclass
class Obstacle:
    type: str
    x: float
    y: float
    w: float
    h: float
    distance_est: float
    lane_hint: str
    confidence: float


def load_scene(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        scene = json.load(fh)
    validate_scene(scene)
    return scene


def validate_scene(scene: dict) -> None:
    required = {"frame_id", "obstacles", "free_space", "ego_state"}
    missing = required - scene.keys()
    if missing:
        raise ValueError(f"missing required keys: {sorted(missing)}")

    for key in ("left", "center", "right"):
        if key not in scene["free_space"]:
            raise ValueError(f"free_space missing key: {key}")

    for key in ("speed_mps", "speed_level", "heading"):
        if key not in scene["ego_state"]:
            raise ValueError(f"ego_state missing key: {key}")

    for idx, item in enumerate(scene["obstacles"]):
        for key in ("type", "x", "y", "w", "h", "distance_est", "lane_hint", "confidence"):
            if key not in item:
                raise ValueError(f"obstacles[{idx}] missing key: {key}")


def to_obstacles(scene: dict) -> list[Obstacle]:
    return [Obstacle(**item) for item in scene["obstacles"]]


def decide(scene: dict) -> dict:
    free = scene["free_space"]
    ego = scene["ego_state"]
    obstacles = to_obstacles(scene)

    center_obstacles = [o for o in obstacles if o.lane_hint == "center"]
    nearest = min((o.distance_est for o in obstacles), default=999.0)
    nearest_center = min((o.distance_est for o in center_obstacles), default=999.0)

    emergency_stop_m = 0.8
    stop_m = 1.2
    slow_m = 2.0
    center_blocked = free["center"] < 0.25 or nearest_center < 1.8
    left_better = free["left"] - free["right"] > 0.12
    right_better = free["right"] - free["left"] > 0.12

    if nearest < emergency_stop_m:
        return output(ACTIONS["stop"], 0.98, f"obstacle within emergency distance {nearest:.2f}m")

    if center_blocked and free["left"] < 0.25 and free["right"] < 0.25:
        if ego["speed_level"] in {"medium", "fast"} or nearest_center < stop_m:
            return output(ACTIONS["stop"], 0.93, "center blocked and both sides unsafe")
        return output(ACTIONS["wait"], 0.81, "center blocked and side corridors too narrow")

    if nearest_center < slow_m and center_blocked:
        if left_better and free["left"] > 0.40:
            return output(ACTIONS["left"], confidence_from_gap(free["left"], nearest_center),
                          "center obstacle ahead, left corridor safer")
        if right_better and free["right"] > 0.40:
            return output(ACTIONS["right"], confidence_from_gap(free["right"], nearest_center),
                          "center obstacle ahead, right corridor safer")
        return output(ACTIONS["slow"], 0.72, "center obstacle ahead, reducing speed while searching path")

    if free["center"] >= 0.45 and nearest > slow_m:
        return output(ACTIONS["keep"], 0.88, "center corridor is clear")

    if free["left"] > 0.55 and left_better:
        return output(ACTIONS["left"], 0.70, "left corridor wider than center and right")

    if free["right"] > 0.55 and right_better:
        return output(ACTIONS["right"], 0.70, "right corridor wider than center and left")

    return output(ACTIONS["slow"], 0.60, "scene ambiguous, applying conservative fallback")


def confidence_from_gap(free_space: float, distance_m: float) -> float:
    confidence = 0.55 + min(free_space, 0.9) * 0.25 + min(max(distance_m, 0.0), 3.0) * 0.06
    return round(min(confidence, 0.95), 2)


def output(action: str, confidence: float, reason: str) -> dict:
    return {
        "action": action,
        "confidence": round(confidence, 2),
        "reason": reason,
        "source": "rule_engine",
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <scene.json>", file=sys.stderr)
        return 2

    scene = load_scene(Path(argv[1]))
    decision = decide(scene)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
