#!/usr/bin/env python3
"""
Phase A4 v2: TinyLlama-1.1B structured decision brain (OPTIMIZED)
Input: scene_context JSON → Output: structured action JSON

v2 优化点:
  1. 标签映射 — InceptionV3 ImageNet 标签 → COCO 驾驶语义
  2. 恢复采样解码 — CPU 推理避免 MPS NaN
  3. 重写 prompt — 决策树式规则 + 更丰富的 few-shot
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SUMMARY = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU/phase_a1_board_summary.json")
OUTPUT = os.path.expanduser("~/Documents/端测/输出/行车记录仪-LLM决策结果-v2")
os.makedirs(OUTPUT, exist_ok=True)

# ── ImageNet → 驾驶语义 粗粒度映射 ─────────────────────────
IMAGENET_TO_DRIVING = {
    # Vehicles
    "moving van": "car", "minivan": "car", "limousine": "car", "cab": "car",
    "police van": "car", "pickup": "car", "sports car": "car", "racer": "car",
    "convertible": "car", "beach wagon": "car", "tow truck": "truck",
    "garbage truck": "truck", "trailer truck": "truck", "minibus": "bus",
    "passenger car": "car", "amphibian": "car", "reel": "car",
    "crane": "truck", "snowplow": "truck", "trolleybus": "bus",
    "space shuttle": "truck",
    # Two-wheelers
    "motor scooter": "motorcycle", "tricycle": "bicycle",
    # Traffic items
    "traffic light": "traffic light", "stop sign": "traffic sign",
    # Not driving-relevant → generic
    "cassette player": "unknown", "espresso maker": "unknown",
    "milk can": "unknown", "lotion": "unknown", "dishwasher": "unknown",
    "oil filter": "unknown", "pill bottle": "unknown", "binder": "unknown",
    "cleaver": "unknown", "thresher": "unknown", "book jacket": "unknown",
    "chain saw": "unknown", "screen": "unknown", "ski": "unknown",
    "hand-held computer": "unknown", "desktop computer": "unknown",
    "face powder": "unknown", "Windsor tie": "unknown",
    "container ship": "unknown", "bearskin": "unknown",
    "packet": "unknown", "pole": "unknown", "nipple": "unknown",
    "beaker": "unknown", "rock beauty": "unknown",
    "shoji": "unknown", "slot": "unknown", "space bar": "unknown",
    "guillotine": "unknown", "cassette": "unknown",
    "street sign": "traffic sign", "affenpinscher": "unknown",
}

def map_obstacle_type(label):
    """Map InceptionV3/COCO label to driving-semantic type."""
    if label is None:
        return "unknown"
    # Direct match
    if label in IMAGENET_TO_DRIVING:
        return IMAGENET_TO_DRIVING[label]
    # Case-insensitive
    lower = label.lower()
    for k, v in IMAGENET_TO_DRIVING.items():
        if lower == k.lower():
            return v
    return label  # keep original if unknown


# ── v2 System prompt ──────────────────────────────────────
SYSTEM_PROMPT = """You are a self-driving obstacle-avoidance controller.
Given a scene description as JSON, output ONE safe action.

DECISION LOGIC (apply in this order, top rule wins):

1. EMERGENCY STOP: If ANY obstacle distance < 0.8m, output STOP.
   → {"action":"STOP","confidence":0.98,"reason":"obstacle at X.Xm, too close"}

2. CENTER BLOCKED (center free_space < 0.25):
   - If left free_space > 0.40 → BYPASS_LEFT
   - If right free_space > 0.40 → BYPASS_RIGHT
   - If BOTH sides < 0.25 AND speed is "fast" or "medium" → STOP
   - Otherwise → WAIT

3. CENTER NARROW (center free_space 0.25-0.45):
   - If a side has >0.50 free_space → BYPASS to that side
   - Otherwise → SLOW_DOWN

4. CENTER CLEAR (center free_space > 0.45):
   → KEEP_CENTER

IMPORTANT:
- obstacle type "unknown" = ignore it (not a real obstacle)
- distance_est = 8.0 means very far, don't react
- Free spaces are NOT cumulative — center=0.30 means 30% open, not "partially plus side"

Output ONLY valid JSON, no other text:
{"action":"X","confidence":0.XX,"reason":"one sentence explanation"}"""

# ── v2 Few-shot examples (covering all 6 actions + edge cases) ──
FEW_SHOT = [
    # 1. Emergency STOP
    (
        '{"obstacles":[{"type":"car","distance_est":0.5,"lane_hint":"center"}],"free_space":{"left":0.90,"center":0.10,"right":0.85},"ego_state":{"speed_level":"fast"}}',
        '{"action":"STOP","confidence":0.98,"reason":"obstacle at 0.5m in center lane, emergency stop"}'
    ),
    # 2. Center blocked, left clear → BYPASS_LEFT
    (
        '{"obstacles":[{"type":"car","distance_est":1.4,"lane_hint":"center"},{"type":"truck","distance_est":2.0,"lane_hint":"center"}],"free_space":{"left":0.75,"center":0.12,"right":0.30},"ego_state":{"speed_level":"slow"}}',
        '{"action":"BYPASS_LEFT","confidence":0.85,"reason":"center blocked at 0.12, left corridor 0.75 is wide open"}'
    ),
    # 3. Center blocked, right clear → BYPASS_RIGHT
    (
        '{"obstacles":[{"type":"car","distance_est":1.8,"lane_hint":"center"},{"type":"bus","distance_est":2.5,"lane_hint":"center"}],"free_space":{"left":0.25,"center":0.10,"right":0.65},"ego_state":{"speed_level":"slow"}}',
        '{"action":"BYPASS_RIGHT","confidence":0.83,"reason":"center blocked at 0.10, right corridor 0.65 is open"}'
    ),
    # 4. Center narrow → SLOW_DOWN
    (
        '{"obstacles":[{"type":"car","distance_est":2.0,"lane_hint":"center"},{"type":"car","distance_est":3.5,"lane_hint":"left"}],"free_space":{"left":0.60,"center":0.35,"right":0.55},"ego_state":{"speed_level":"medium"}}',
        '{"action":"SLOW_DOWN","confidence":0.70,"reason":"center narrow at 0.35, reduce speed to assess"}'
    ),
    # 5. Wide open highway → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"left"},{"type":"truck","distance_est":8.0,"lane_hint":"right"}],"free_space":{"left":0.55,"center":0.92,"right":0.50},"ego_state":{"speed_level":"fast"}}',
        '{"action":"KEEP_CENTER","confidence":0.94,"reason":"center wide open at 0.92, obstacles are far in side lanes"}'
    ),
    # 6. Both sides blocked, fast speed → STOP
    (
        '{"obstacles":[{"type":"car","distance_est":1.2,"lane_hint":"center"},{"type":"truck","distance_est":1.0,"lane_hint":"left"},{"type":"bus","distance_est":1.5,"lane_hint":"right"}],"free_space":{"left":0.15,"center":0.08,"right":0.12},"ego_state":{"speed_level":"fast"}}',
        '{"action":"STOP","confidence":0.96,"reason":"all lanes blocked, speed is fast, must stop"}'
    ),
    # 7. Center blocked both sides marginal, slow → WAIT
    (
        '{"obstacles":[{"type":"car","distance_est":1.5,"lane_hint":"center"}],"free_space":{"left":0.22,"center":0.10,"right":0.20},"ego_state":{"speed_level":"slow"}}',
        '{"action":"WAIT","confidence":0.75,"reason":"center blocked, both sides too narrow, waiting for clearance"}'
    ),
    # 8. Center clear with nearby obstacle but in different lane → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":3.0,"lane_hint":"left"},{"type":"car","distance_est":4.0,"lane_hint":"right"}],"free_space":{"left":0.50,"center":0.88,"right":0.45},"ego_state":{"speed_level":"medium"}}',
        '{"action":"KEEP_CENTER","confidence":0.90,"reason":"center corridor 0.88 is clear, obstacles in side lanes only"}'
    ),
    # 9. Many obstacles but center still open → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":5.0,"lane_hint":"left"},{"type":"car","distance_est":6.0,"lane_hint":"left"},{"type":"truck","distance_est":4.0,"lane_hint":"right"},{"type":"bus","distance_est":7.0,"lane_hint":"right"}],"free_space":{"left":0.40,"center":0.82,"right":0.38},"ego_state":{"speed_level":"medium"}}',
        '{"action":"KEEP_CENTER","confidence":0.88,"reason":"center 0.82 is open despite many obstacles in side lanes"}'
    ),
    # 10. Narrow center but left much wider → BYPASS_LEFT
    (
        '{"obstacles":[{"type":"car","distance_est":2.2,"lane_hint":"center"}],"free_space":{"left":0.72,"center":0.32,"right":0.35},"ego_state":{"speed_level":"slow"}}',
        '{"action":"BYPASS_LEFT","confidence":0.78,"reason":"center narrow at 0.32, left 0.72 is much wider"}'
    ),
]


# ── Model loading ────────────────────────────────────────
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load on MPS GPU — use float16 + greedy decoding to avoid NaN bug
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # GPU works better with float16
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    print(f"  Model loaded on MPS GPU (float16)")
    return tokenizer, model


def build_prompt(scene_json: str) -> list:
    """Build a few-shot chat prompt for TinyLlama Chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, asst_msg in FEW_SHOT:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": scene_json})
    return messages


def llm_decide(tokenizer, model, scene_json: str) -> dict:
    """Send scene to TinyLlama, parse JSON output."""
    messages = build_prompt(scene_json)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,         # Greedy: avoids MPS NaN, deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    result_text = result_text.strip()

    # Extract JSON
    try:
        start = result_text.find("{")
        end = result_text.rfind("}")
        if start >= 0 and end > start:
            json_str = result_text[start:end+1]
            parsed = json.loads(json_str)
            # Validate action
            valid_actions = {"STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"}
            if parsed.get("action") not in valid_actions:
                parsed["action"] = "SLOW_DOWN"
            return parsed
    except json.JSONDecodeError:
        pass

    return {
        "action": "SLOW_DOWN", "confidence": 0.50,
        "reason": "LLM parse error", "raw_output": result_text[:100]
    }


# ── Main ──────────────────────────────────────────────────
def main():
    if not os.path.exists(SUMMARY):
        print(f"ERROR: {SUMMARY} not found. Run phase_a1_board.py first.", file=sys.stderr)
        return 1

    with open(SUMMARY) as f:
        results = json.load(f)

    tokenizer, model = load_model()

    all_results = []
    actions_llm = Counter()
    actions_rule = Counter()
    t0 = time.time()

    frames_with_obs = [r for r in results if r["scene_context"].get("obstacles")]
    print(f"\nProcessing {len(frames_with_obs)} frames with obstacles ({len(results)} total)\n")

    for idx, result in enumerate(frames_with_obs):
        scene = result["scene_context"]
        decision_rule = result["decision"]

        # Build clean scene JSON — use YOLO label (driving semantic),
        # not InceptionV3 label (ImageNet gibberish)
        obstacles_clean = []
        for o in scene.get("obstacles", []):
            raw_type = o.get("yolo_label") or o.get("type", "unknown")
            obstacles_clean.append({
                "type": map_obstacle_type(raw_type),
                "distance_est": o["distance_est"],
                "lane_hint": o["lane_hint"],
            })

        scene_clean = {
            "obstacles": obstacles_clean,
            "free_space": scene["free_space"],
            "ego_state": scene["ego_state"],
        }
        scene_json = json.dumps(scene_clean, ensure_ascii=False)

        print(f"[{idx+1}/{len(frames_with_obs)}] Frame {result['frame_id']} …", end=" ", flush=True)
        t1 = time.time()
        decision_llm = llm_decide(tokenizer, model, scene_json)
        t_llm = (time.time() - t1) * 1000

        action_llm = decision_llm.get("action", "SLOW_DOWN")
        confidence_llm = decision_llm.get("confidence", 0)
        action_rule = decision_rule.get("action", "N/A")

        match = "✅" if action_llm == action_rule else "⚠️"
        print(f"{match} LLM: {action_llm}({confidence_llm:.2f}) vs Rule: {action_rule} [{t_llm:.0f}ms]", flush=True)

        actions_llm[action_llm] += 1
        actions_rule[action_rule] += 1

        all_results.append({
            "frame_idx": result["frame_id"],
            "scene": scene_clean,
            "decision_llm": decision_llm,
            "decision_rule": decision_rule,
            "llm_runtime_ms": round(t_llm),
            "match": action_llm == action_rule,
        })

    total_t = time.time() - t0
    print(f"\n{'='*60}")

    n = len(all_results)
    matches = sum(1 for r in all_results if r["match"])
    print(f"LLM decisions: {n} frames ({total_t:.1f}s, avg {total_t/n*1000:.0f}ms/frame)")
    print(f"Agreement with rule engine: {matches}/{n} ({matches/n*100:.0f}%)")
    print(f"\nDecision distribution:")
    for act in ["STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"]:
        llm_cnt = actions_llm.get(act, 0)
        rule_cnt = actions_rule.get(act, 0)
        bar = "█" * llm_cnt
        print(f"  {act:14s} LLM={llm_cnt:3d}  Rule={rule_cnt:3d}  {bar}")

    # Show disagreements
    disagreements = [r for r in all_results if not r["match"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements[:20]:  # Show first 20
            scene = r["scene"]
            obs_types = list(set(o["type"] for o in scene.get("obstacles", [])))
            llm_act = r["decision_llm"]["action"]
            rule_act = r["decision_rule"]["action"]
            llm_reason = r["decision_llm"].get("reason", "")[:60]
            print(f"  Frame {r['frame_idx']}: LLM={llm_act} vs Rule={rule_act} | obs={obs_types} | {llm_reason}")
        if len(disagreements) > 20:
            print(f"  ... and {len(disagreements)-20} more")

    # Save
    out_path = os.path.join(OUTPUT, "phase_a4_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
