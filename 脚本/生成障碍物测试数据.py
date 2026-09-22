#!/usr/bin/env python3
"""
合成障碍物测试数据生成器 v2
=======================
在 KITTI 道路背景上叠加绘制的障碍物精灵（车/人/锥桶/护栏/卡车），
为每一种避障决策动作生成多帧变体，同时输出 ground-truth scene_context.json。

关键修复 (v2): 语义字段（distance_est / lane_hint / free_space）基于场景意图
计算，不依赖被裁剪的视觉 bbox 面积，确保规则引擎能精确触发预期决策。

决策动作覆盖（来自 rule_engine.py 的 6 条路径）：
  STOP, WAIT, SLOW_DOWN, BYPASS_LEFT, BYPASS_RIGHT, KEEP_CENTER

输出结构:
  测试数据/障碍物合成测试帧/
    ├── frames/          # 合成 PNG 帧
    ├── scenes/          # 每帧的 scene_context.json
    ├── scenarios.json   # 场景索引 (文件名 → 预期动作)
    └── manifest.json    # 完整元数据
"""

import json
import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── 路径配置 ──────────────────────────────────────────────
BASE = Path(os.path.expanduser("~/Documents/端测"))
KITTI_POOL = [
    BASE / "测试数据/KITTI-住宅区84帧",
    BASE / "测试数据/KITTI-城市道路131帧",
    BASE / "测试数据/KITTI-高速场景28帧",
]
OUTPUT = BASE / "测试数据/障碍物合成测试帧"
FRAMES_DIR = OUTPUT / "frames"
SCENES_DIR = OUTPUT / "scenes"

FRAME_W, FRAME_H = 640, 640
HORIZON_Y = int(FRAME_H * 0.38)
MAX_DISTANCE_M = 35.0


# ═══════════════════════════════════════════════════════════
# 障碍物精灵工厂
# ═══════════════════════════════════════════════════════════

def _random_vehicle_color() -> Tuple[int, int, int]:
    return random.choice([
        (200, 200, 210), (40, 40, 55), (210, 60, 50),
        (25, 90, 180), (240, 240, 240), (50, 50, 50),
    ])


def create_car_sprite(size_ratio: float = 1.0) -> np.ndarray:
    bw, bh = int(220 * size_ratio), int(100 * size_ratio)
    img = np.zeros((bh, bw, 4), dtype=np.uint8)
    body_c = _random_vehicle_color()
    roof_top, floor_y = bh // 3, bh - 6
    cv2.rectangle(img, (8, roof_top), (bw - 8, floor_y), (*body_c, 255), -1)
    cv2.rectangle(img, (6, roof_top + 5), (bw - 6, floor_y - 2), (*body_c, 255), -1)
    win_l, win_r, win_bot = bw // 5, 4 * bw // 5, bh // 2 + 4
    cv2.rectangle(img, (win_l, roof_top - 3), (win_r, win_bot), (180, 210, 240, 255), -1)
    cv2.line(img, (bw // 2, roof_top - 3), (bw // 2 + 3, win_bot), (*body_c, 255), 3)
    r = bh // 8
    for wx in [bw // 4, 3 * bw // 4]:
        cv2.circle(img, (wx, floor_y + r - 2), r, (25, 25, 25, 255), -1)
        cv2.circle(img, (wx, floor_y + r - 2), r // 2, (80, 80, 80, 255), -1)
    cv2.circle(img, (bw - 10, bh // 3), 4, (240, 240, 200, 255), -1)
    cv2.circle(img, (10, bh // 3), 4, (255, 80, 40, 255), -1)
    cv2.rectangle(img, (4, floor_y), (bw - 4, bh - 2), (60, 60, 60, 255), -1)
    return img


def create_truck_sprite(size_ratio: float = 1.0) -> np.ndarray:
    bw, bh = int(300 * size_ratio), int(130 * size_ratio)
    img = np.zeros((bh, bw, 4), dtype=np.uint8)
    body_c = _random_vehicle_color()
    floor_y = bh - 8
    cv2.rectangle(img, (5, bh // 3), (bw - 6, floor_y), (*body_c, 255), -1)
    cab_l, cab_r = 3 * bw // 5, bw - 10
    cv2.rectangle(img, (cab_l, bh // 4 - 2), (cab_r, bh // 3 + 4), (*body_c, 255), -1)
    cv2.rectangle(img, (int(cab_l * 1.02), bh // 4), (cab_r - 5, bh // 3 + 2), (190, 220, 250, 255), -1)
    r = bh // 9
    for wx in [bw // 4, 7 * bw // 10, 8 * bw // 10, 9 * bw // 10]:
        cv2.circle(img, (wx, floor_y + r - 2), r, (25, 25, 25, 255), -1)
    cv2.rectangle(img, (4, floor_y), (bw - 4, bh - 2), (50, 50, 50, 255), -1)
    return img


def create_person_sprite(size_ratio: float = 1.0) -> np.ndarray:
    pw, ph = int(40 * size_ratio), int(140 * size_ratio)
    img = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx = pw // 2
    head_r = pw // 3
    cv2.circle(img, (cx, head_r + 2), head_r, (180, 150, 130, 255), -1)
    top_c = random.choice([(40, 40, 180), (180, 50, 50), (50, 130, 50), (30, 30, 30), (100, 70, 40)])
    torso_t, torso_b = 2 * head_r + 2, ph * 2 // 3
    cv2.rectangle(img, (pw // 4, torso_t), (3 * pw // 4, torso_b), (*top_c, 255), -1)
    pants = random.choice([(20, 20, 60), (40, 40, 40)])
    for lx in [pw // 4, pw // 2 + 1]:
        cv2.rectangle(img, (lx, torso_b), (lx + pw // 4 - 1, ph - 3), (*pants, 255), -1)
    for lx in [pw // 4, pw // 2 + 1]:
        cv2.rectangle(img, (lx - 1, ph - 5), (lx + pw // 4, ph - 1), (50, 50, 50, 255), -1)
    return img


def create_cone_sprite(size_ratio: float = 1.0) -> np.ndarray:
    cw, ch = int(36 * size_ratio), int(80 * size_ratio)
    img = np.zeros((ch, cw, 4), dtype=np.uint8)
    cx, base_y = cw // 2, ch - 8
    pts = np.array([[cx, 3], [4, base_y], [cw - 4, base_y]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 100, 255, 255))
    for y_frac in [0.35, 0.65]:
        yy = int(3 + (base_y - 3) * y_frac)
        half_w = int(cx * (1 - y_frac))
        stripe = np.array([
            [cx - half_w, yy - 2], [cx + half_w, yy - 2],
            [cx + half_w + 1, yy + 2], [cx - half_w - 1, yy + 2],
        ], dtype=np.int32)
        cv2.fillPoly(img, [stripe], (255, 255, 255, 255))
    cv2.rectangle(img, (2, base_y), (cw - 2, ch - 1), (0, 100, 255, 255), -1)
    return img


def create_barrier_sprite(size_ratio: float = 1.0) -> np.ndarray:
    bw, bh = int(200 * size_ratio), int(55 * size_ratio)
    img = np.zeros((bh, bw, 4), dtype=np.uint8)
    stripe_w = max(bw // 10, 12)
    for i in range(bw // stripe_w + 1):
        color = ((230, 60, 50, 255), (250, 250, 250, 255))[i % 2]
        cv2.rectangle(img, (i * stripe_w, bh // 3), (i * stripe_w + stripe_w, bh), color, -1)
    cv2.rectangle(img, (0, 3), (bw, bh // 3 + 3), (100, 100, 100, 255), -1)
    return img


SPRITE_FACTORY = {
    "car": create_car_sprite, "truck": create_truck_sprite,
    "person": create_person_sprite, "cone": create_cone_sprite,
    "barrier": create_barrier_sprite,
}

SPRITE_REF_DISTANCE = {
    "car": 6.0, "truck": 7.0, "person": 3.5, "cone": 3.0, "barrier": 7.0,
}

# ── 类型宽度因子 (ground-truth free_space 计算用) ──
OBJ_WIDTH_FACTOR = {
    "car": 0.38, "truck": 0.55, "barrier": 0.42,
    "person": 0.18, "cone": 0.14,
}


# ═══════════════════════════════════════════════════════════
# 场景规格
# ═══════════════════════════════════════════════════════════

@dataclass
class ObstacleSpec:
    obj_type: str          # car | truck | person | cone | barrier
    lane: str              # left | center | right
    distance_m: float      # 用于视觉缩放/定位
    lateral_shift: float = 0.0
    dist_override: float = None  # 覆盖 scene JSON 中的 distance_est (None=用 distance_m)


@dataclass
class ScenarioSpec:
    name: str
    expected_action: str
    obstacles: List[ObstacleSpec]
    ego_speed_mps: float = 2.5
    heading: str = "forward"
    description: str = ""
    free_left: float = None    # 覆盖 scene JSON free_space.left
    free_center: float = None  # 覆盖 scene JSON free_space.center
    free_right: float = None   # 覆盖 scene JSON free_space.right


SCENARIOS: List[ScenarioSpec] = [
    # ═══ STOP ═══
    # 规则路径: nearest < 0.8 → STOP (emergency)
    ScenarioSpec("stop_01_emergency", "STOP",
        [ObstacleSpec("car", "center", 0.6)],
        ego_speed_mps=2.0, free_left=1.0, free_center=0.0, free_right=1.0,
        description="Emergency: car at 0.6m center → nearest < 0.8 → STOP"),

    # 规则路径: center_blocked + both sides < 0.25 + speed medium/fast → STOP
    ScenarioSpec("stop_02_fully_blocked", "STOP",
        [ObstacleSpec("truck", "center", 0.9, dist_override=0.85),
         ObstacleSpec("car", "left", 1.2, dist_override=1.10),
         ObstacleSpec("barrier", "right", 0.8, dist_override=0.75)],
        ego_speed_mps=3.5, free_left=0.15, free_center=0.10, free_right=0.12,
        description="Fully blocked all lanes, fast speed → STOP"),

    # ═══ WAIT ═══
    # 规则路径: center_blocked + both sides < 0.25 + speed slow → WAIT
    ScenarioSpec("wait_01_narrow_all", "WAIT",
        [ObstacleSpec("barrier", "center", 1.3, dist_override=1.60),
         ObstacleSpec("barrier", "left", 1.5, dist_override=1.50),
         ObstacleSpec("barrier", "right", 1.6, dist_override=1.55)],
        ego_speed_mps=0.8, free_left=0.18, free_center=0.15, free_right=0.20,
        description="All corridors narrow but not emergency, slow → WAIT"),

    ScenarioSpec("wait_02_center_and_sides", "WAIT",
        [ObstacleSpec("truck", "center", 1.1, dist_override=1.60),
         ObstacleSpec("cone", "left", 1.4, dist_override=1.50, lateral_shift=-0.2),
         ObstacleSpec("cone", "right", 1.4, dist_override=1.50, lateral_shift=0.2)],
        ego_speed_mps=1.0, free_left=0.22, free_center=0.12, free_right=0.22,
        description="Center blocked, side gaps marginal, slow → WAIT"),

    # ═══ BYPASS_LEFT ═══
    # 规则路径: nearest_center < 2.0 + center_blocked + left_better + left > 0.40
    ScenarioSpec("bypass_left_01_car_center", "BYPASS_LEFT",
        [ObstacleSpec("car", "center", 1.8, dist_override=1.75, lateral_shift=0.1)],
        ego_speed_mps=2.0, free_left=0.85, free_center=0.18, free_right=0.70,
        description="Car in center at 1.75m, left wider → BYPASS_LEFT"),

    ScenarioSpec("bypass_left_02_truck_right", "BYPASS_LEFT",
        [ObstacleSpec("truck", "center", 2.5, dist_override=1.70, lateral_shift=0.15),
         ObstacleSpec("car", "right", 3.0, dist_override=3.0)],
        ego_speed_mps=2.5, free_left=0.80, free_center=0.20, free_right=0.50,
        description="Truck center + car right, only left open → BYPASS_LEFT"),

    ScenarioSpec("bypass_left_03_wide_left", "BYPASS_LEFT",
        [ObstacleSpec("barrier", "center", 1.9, dist_override=1.75, lateral_shift=0.2),
         ObstacleSpec("barrier", "right", 2.5, dist_override=2.0)],
        ego_speed_mps=1.5, free_left=0.75, free_center=0.20, free_right=0.35,
        description="Center+right blocked, wide left → BYPASS_LEFT"),

    ScenarioSpec("bypass_left_04_cone_center", "BYPASS_LEFT",
        [ObstacleSpec("cone", "center", 1.7, dist_override=1.70, lateral_shift=0.05)],
        ego_speed_mps=1.8, free_left=0.78, free_center=0.22, free_right=0.65,
        description="Cone in center, left naturally wider → BYPASS_LEFT"),

    # ═══ BYPASS_RIGHT ═══
    ScenarioSpec("bypass_right_01_car_left", "BYPASS_RIGHT",
        [ObstacleSpec("car", "center", 1.8, dist_override=1.75, lateral_shift=-0.1),
         ObstacleSpec("car", "left", 2.2, dist_override=1.60)],
        ego_speed_mps=2.0, free_left=0.38, free_center=0.20, free_right=0.82,
        description="Cars center+left, right open → BYPASS_RIGHT"),

    ScenarioSpec("bypass_right_02_wide_right", "BYPASS_RIGHT",
        [ObstacleSpec("barrier", "center", 1.9, dist_override=1.75, lateral_shift=-0.2),
         ObstacleSpec("barrier", "left", 2.5, dist_override=2.0)],
        ego_speed_mps=1.5, free_left=0.35, free_center=0.20, free_right=0.78,
        description="Center+left blocked, wide right → BYPASS_RIGHT"),

    ScenarioSpec("bypass_right_03_truck_center", "BYPASS_RIGHT",
        [ObstacleSpec("truck", "center", 2.8, dist_override=1.70, lateral_shift=-0.1)],
        ego_speed_mps=2.2, free_left=0.52, free_center=0.20, free_right=0.80,
        description="Truck center-left, right much wider → BYPASS_RIGHT"),

    ScenarioSpec("bypass_right_04_person_center", "BYPASS_RIGHT",
        [ObstacleSpec("person", "center", 1.6, dist_override=1.70, lateral_shift=-0.15)],
        ego_speed_mps=1.5, free_left=0.52, free_center=0.22, free_right=0.78,
        description="Pedestrian center-left, bypass right → BYPASS_RIGHT"),

    # ═══ SLOW_DOWN ═══
    # 规则路径: nearest_center < 2.0 + center_blocked, 但两侧都不够好 → SLOW_DOWN
    ScenarioSpec("slow_01_center_ambiguous", "SLOW_DOWN",
        [ObstacleSpec("car", "center", 1.5, dist_override=1.75),
         ObstacleSpec("cone", "left", 2.0, dist_override=2.0, lateral_shift=-0.3),
         ObstacleSpec("cone", "right", 2.0, dist_override=2.0, lateral_shift=0.3)],
        ego_speed_mps=2.0, free_left=0.50, free_center=0.20, free_right=0.50,
        description="Center obstacle, both sides equally narrow → SLOW_DOWN"),

    ScenarioSpec("slow_02_narrow_passage", "SLOW_DOWN",
        [ObstacleSpec("barrier", "center", 1.4, dist_override=1.70),
         ObstacleSpec("barrier", "left", 1.8, dist_override=2.0),
         ObstacleSpec("barrier", "right", 2.0, dist_override=2.0)],
        ego_speed_mps=2.5, free_left=0.42, free_center=0.18, free_right=0.42,
        description="All lanes partially blocked → SLOW_DOWN"),

    # 规则路径: nearest <= 2.0 使 step4 不匹配，fallback 到 step7 SLOW_DOWN
    ScenarioSpec("slow_03_ambiguous_fallback", "SLOW_DOWN",
        [ObstacleSpec("car", "left", 2.5, dist_override=1.90, lateral_shift=0.2),
         ObstacleSpec("car", "right", 2.5, dist_override=1.90, lateral_shift=-0.2)],
        ego_speed_mps=3.0, free_left=0.48, free_center=0.50, free_right=0.48,
        description="Obstacles on sides, nearest < 2.0 → bypass step4 → SLOW_DOWN"),

    # ═══ KEEP_CENTER ═══
    # 规则路径: center >= 0.45 + nearest > 2.0
    ScenarioSpec("keep_01_clear_road", "KEEP_CENTER",
        [], ego_speed_mps=3.0, free_left=1.0, free_center=1.0, free_right=1.0,
        description="Clear road ahead → KEEP_CENTER"),

    ScenarioSpec("keep_02_distant_cars", "KEEP_CENTER",
        [ObstacleSpec("car", "center", 8.0),
         ObstacleSpec("car", "left", 9.0)],
        ego_speed_mps=3.5, free_left=0.85, free_center=0.82, free_right=1.0,
        description="Distant cars, center still free → KEEP_CENTER"),

    ScenarioSpec("keep_03_far_person_side", "KEEP_CENTER",
        [ObstacleSpec("person", "left", 6.5, dist_override=6.5, lateral_shift=-0.3)],
        ego_speed_mps=2.0, free_left=0.92, free_center=1.0, free_right=1.0,
        description="Pedestrian far left sidewalk → KEEP_CENTER"),
]


# ═══════════════════════════════════════════════════════════
# 合成引擎
# ═══════════════════════════════════════════════════════════

def load_kitti_backgrounds() -> List[Tuple[Path, str]]:
    all_frames = []
    for pool_dir in KITTI_POOL:
        if not pool_dir.exists():
            continue
        scene_type = pool_dir.name.replace("KITTI-", "").replace("场景", "").replace("帧", "")
        all_frames.extend((p, scene_type) for p in sorted(pool_dir.glob("*.png")))
    random.shuffle(all_frames)
    return all_frames


def lane_to_pixel_x(lane: str, shift: float) -> int:
    base = {"left": FRAME_W * 0.22, "center": FRAME_W * 0.50, "right": FRAME_W * 0.78}
    jitter = random.uniform(-0.03, 0.03) * FRAME_W
    return int(base[lane] + shift * FRAME_W * 0.25 + jitter)


def distance_to_y_bottom(distance_m: float) -> int:
    frac = min(max(distance_m, 0.1) / MAX_DISTANCE_M, 1.0)
    return int(HORIZON_Y + (FRAME_H - HORIZON_Y) * (1.0 - frac ** 0.7))


def distance_to_scale(obj_type: str, distance_m: float) -> float:
    ref_d = SPRITE_REF_DISTANCE.get(obj_type, 5.0)
    return float(np.clip(ref_d / max(distance_m, 0.3), 0.08, 3.5))


def overlay_sprite(bg_rgb: np.ndarray, sprite_rgba: np.ndarray,
                   center_x: int, bottom_y: int) -> np.ndarray:
    sh, sw = sprite_rgba.shape[:2]
    bh, bw = bg_rgb.shape[:2]

    left, top = center_x - sw // 2, bottom_y - sh
    roi_left = max(left, 0)
    roi_top = max(top, 0)
    roi_right = min(left + sw, bw)
    roi_bottom = min(bottom_y, bh)

    if roi_right <= roi_left or roi_bottom <= roi_top:
        return bg_rgb

    roi = bg_rgb[roi_top:roi_bottom, roi_left:roi_right]
    sprite_crop = sprite_rgba[roi_top - top:roi_bottom - top,
                               roi_left - left:roi_right - left]
    alpha = sprite_crop[:, :, 3:4] / 255.0
    roi[:] = (sprite_crop[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return bg_rgb


def composite_frame(bg_path: Path, spec: ScenarioSpec,
                    frame_id: int, rng_seed: int) -> Tuple[np.ndarray, dict]:
    """
    合成一帧 + 生成 ground-truth scene_context。
    
    **关键设计**: 语义字段 (distance_est / lane_hint / free_space) 基于
    场景意图计算，不依赖被裁剪的视觉 bbox 面积。这样才能在规则引擎中
    精确触发预期决策。
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    bg = cv2.imread(str(bg_path))
    if bg is None:
        raise FileNotFoundError(f"Cannot read background: {bg_path}")
    bg = cv2.resize(bg, (FRAME_W, FRAME_H))
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    obstacle_records = []
    lane_occupancy = {"left": 0.0, "center": 0.0, "right": 0.0}

    for obs in spec.obstacles:
        cx = lane_to_pixel_x(obs.lane, obs.lateral_shift)
        by = distance_to_y_bottom(obs.distance_m)
        scale = distance_to_scale(obs.obj_type, obs.distance_m)

        sprite_fn = SPRITE_FACTORY.get(obs.obj_type)
        if sprite_fn is None:
            continue
        sprite = sprite_fn(scale)
        sh, sw = sprite.shape[:2]

        # 叠加精灵到背景
        bg = overlay_sprite(bg, sprite, cx, by)

        # ── 视觉 bbox (clip 到图像边界) ──
        left_px = max(0, cx - sw // 2)
        top_px = max(0, by - sh)
        right_px = min(FRAME_W, left_px + sw)
        bot_px = min(FRAME_H, by)

        norm_x = (left_px + right_px) / 2.0 / FRAME_W
        norm_y = (top_px + bot_px) / 2.0 / FRAME_H
        norm_w = max(right_px - left_px, 1) / FRAME_W
        norm_h = max(bot_px - top_px, 1) / FRAME_H

        # ── 语义字段: 基于场景意图 (支持 override) ──
        if obs.dist_override is not None:
            distance_est = round(obs.dist_override, 2)
        else:
            dist_noise = random.uniform(-0.06, 0.06) * obs.distance_m
            distance_est = round(max(0.1, obs.distance_m + dist_noise), 2)
        lane_hint = obs.lane

        obstacle_records.append({
            "type": obs.obj_type,
            "x": round(norm_x, 3),
            "y": round(norm_y, 3),
            "w": round(norm_w, 3),
            "h": round(norm_h, 3),
            "distance_est": distance_est,
            "lane_hint": lane_hint,
            "confidence": round(random.uniform(0.82, 0.97), 3),
        })

            # ── lane occupancy (仅当未显式指定 free_space 时使用) ──
        w_factor = OBJ_WIDTH_FACTOR.get(obs.obj_type, 0.35)
        proximity = max(0.0, 1.0 - obs.distance_m / 6.0)
        lane_occupancy[obs.lane] += w_factor * proximity

    # free_space: 优先使用 spec 中显式指定的值
    if spec.free_left is not None:
        free_space = {"left": spec.free_left, "center": spec.free_center, "right": spec.free_right}
    else:
        free_space = {
            name: round(max(0.0, min(1.0, 1.0 - min(occ, 0.98))), 2)
            for name, occ in lane_occupancy.items()
        }

    spd = spec.ego_speed_mps
    speed_level = "stop" if spd == 0 else "slow" if spd < 1.5 else "medium" if spd < 4.0 else "fast"

    scene = {
        "frame_id": frame_id,
        "obstacles": obstacle_records,
        "free_space": free_space,
        "ego_state": {
            "speed_mps": round(spd, 2),
            "speed_level": speed_level,
            "heading": spec.heading,
        },
    }

    return bg, scene


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    kitti_bg = load_kitti_backgrounds()
    if not kitti_bg:
        print("ERROR: 未找到 KITTI 背景帧，请检查测试数据目录", file=sys.stderr)
        return 1

    print(f"背景池: {len(kitti_bg)} 帧 (来自 {len(KITTI_POOL)} 个目录)")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    bg_idx = 0
    frame_id = 0

    for si, spec in enumerate(SCENARIOS):
        for variant in range(2):
            fname = f"{spec.name}_v{variant}"
            bg_path, bg_type = kitti_bg[bg_idx % len(kitti_bg)]
            bg_idx += 1
            seed = si * 100 + variant

            try:
                frame_rgb, scene = composite_frame(bg_path, spec, frame_id, seed)
            except Exception as e:
                print(f"  WARN: {fname} 合成失败: {e}")
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(FRAMES_DIR / f"{fname}.png"), frame_bgr)

            scene_path = SCENES_DIR / f"{fname}.json"
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)

            manifest.append({
                "filename": f"{fname}.png",
                "frame_id": frame_id,
                "scene_json": str(scene_path.relative_to(OUTPUT)),
                "expected_action": spec.expected_action,
                "background_type": bg_type,
                "background_frame": bg_path.name,
                "description": spec.description,
                "ego_speed_mps": spec.ego_speed_mps,
                "num_obstacles": len(spec.obstacles),
                "obstacle_types": [o.obj_type for o in spec.obstacles],
            })

            print(f"  [{frame_id:03d}] {fname}.png  →  expect {spec.expected_action}")
            frame_id += 1

    # manifest.json
    with open(OUTPUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # scenarios.json (兼容原有格式)
    with open(OUTPUT / "scenarios.json", "w", encoding="utf-8") as f:
        json.dump([{
            "name": Path(m["filename"]).stem,
            "expect": m["expected_action"],
            "desc": m["description"],
        } for m in manifest], f, ensure_ascii=False, indent=2)

    from collections import Counter
    action_counts = Counter(m["expected_action"] for m in manifest)
    print(f"\n生成完毕: {len(manifest)} 帧 → {FRAMES_DIR}")
    print("动作分布:")
    for act, cnt in action_counts.most_common():
        print(f"  {act}: {cnt} 帧")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
