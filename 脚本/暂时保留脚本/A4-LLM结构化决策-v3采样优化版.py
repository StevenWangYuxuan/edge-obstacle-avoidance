#!/usr/bin/env python3
"""
Phase A4 v3: TinyLlama-1.1B structured decision brain (SAMPLING OPTIMIZED)
Input: scene_context JSON → Output: structured action JSON

v3 核心修复:
  1. 采样解码 do_sample=True — 打破 v2 贪心解码的 BYPASS_RIGHT 死循环
  2. float16 MPS + NaN catch → 贪心回退（速度快 + 稳定）
  3. Prompt 明确解释 free_space 是比例 (0.0=全堵, 1.0=全开)，不是米数
  4. 精简 few-shot 到 8 个（控制在 2048 token 内）
  5. NaN/解析失败自动回退贪心解码
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
OUTPUT = os.path.expanduser("~/Documents/端测/输出/行车记录仪-LLM决策结果-v3")
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
    # Not driving-relevant → ignore
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
    if label in IMAGENET_TO_DRIVING:
        return IMAGENET_TO_DRIVING[label]
    lower = label.lower()
    for k, v in IMAGENET_TO_DRIVING.items():
        if lower == k.lower():
            return v
    return label


# ── v3 System prompt — 核心修复：解释 free_space 语义 ──────
SYSTEM_PROMPT = """You are a self-driving car obstacle-avoidance controller.
Given a scene as JSON, output ONE safe action.

UNDERSTANDING THE INPUT:
- "free_space": a PROPORTION from 0.0 to 1.0
  0.0 = completely blocked (0% open)
  0.5 = half open (50%)
  1.0 = completely open (100%)
  Example: center=0.95 means "95% clear path ahead" — DRIVE STRAIGHT!
- "distance_est": distance in METERS to the obstacle
  8.0 = very far away, ignore it
- "lane_hint": which lane the obstacle is in

DECISION RULES (first match wins, most important rule at top):

RULE 1 — EMERGENCY STOP:
  IF any obstacle distance_est < 0.8m → STOP
  Reason: obstacle too close to avoid

RULE 2 — CENTER IS CLEAR (> 0.80):
  IF center free_space > 0.80 → MUST output KEEP_CENTER
  NO EXCEPTIONS. Even if side lanes have obstacles.
  0.80+ means the road ahead is wide open.
  Examples: center=0.95 → KEEP_CENTER, center=0.88 → KEEP_CENTER

RULE 3 — CENTER BLOCKED (center < 0.25):
  IF left > 0.40 → BYPASS_LEFT
  IF right > 0.40 → BYPASS_RIGHT
  IF both sides < 0.25 AND speed is "fast" → STOP
  IF both sides < 0.25 AND speed is "slow" → WAIT

RULE 4 — CENTER NARROW (center 0.25-0.80):
  IF left > center + 0.15 → BYPASS_LEFT
  IF right > center + 0.15 → BYPASS_RIGHT
  Otherwise → SLOW_DOWN

IMPORTANT:
- obstacle type "unknown" = not a real obstacle, ignore it
- distance_est = 8.0 means very far, do NOT react
- Free space > 0.80 = SAFE to drive straight, do not bypass!

Output ONLY valid JSON, nothing else:
{"action":"KEEP_CENTER","confidence":0.92,"reason":"center 95% open, safe to proceed"}"""


# ── v3 Few-shot: 8 examples 覆盖全6动作 + 强化KEEP_CENTER ──
FEW_SHOT = [
    # ═══ KEEP_CENTER (3/8 — 最匹配数据) ═══
    # 1. Far car in center, 97% open → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"center"}],"free_space":{"left":1.0,"center":0.97,"right":1.0},"ego_state":{"speed_level":"slow"}}',
        '{"action":"KEEP_CENTER","confidence":0.92,"reason":"center 97% open, obstacle far at 8.0m, drive straight"}'
    ),
    # 2. Obstacles in side lanes far away, center 95% → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"left"},{"type":"truck","distance_est":8.0,"lane_hint":"right"}],"free_space":{"left":0.55,"center":0.95,"right":0.50},"ego_state":{"speed_level":"fast"}}',
        '{"action":"KEEP_CENTER","confidence":0.94,"reason":"center 95% open, obstacles far in side lanes"}'
    ),
    # 3. Single car ahead at 3m, center still 92% → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":3.0,"lane_hint":"center"},{"type":"car","distance_est":4.0,"lane_hint":"left"}],"free_space":{"left":0.55,"center":0.92,"right":0.70},"ego_state":{"speed_level":"medium"}}',
        '{"action":"KEEP_CENTER","confidence":0.90,"reason":"center 92% open, plenty of room"}'
    ),

    # ═══ STOP ═══
    # 4. Emergency: obstacle at 0.5m, center blocked → STOP
    (
        '{"obstacles":[{"type":"car","distance_est":0.5,"lane_hint":"center"}],"free_space":{"left":0.90,"center":0.10,"right":0.85},"ego_state":{"speed_level":"fast"}}',
        '{"action":"STOP","confidence":0.98,"reason":"obstacle at 0.5m in center, emergency stop"}'
    ),

    # ═══ BYPASS ═══
    # 5. Center only 12%, left 75% → BYPASS_LEFT
    (
        '{"obstacles":[{"type":"car","distance_est":1.4,"lane_hint":"center"},{"type":"truck","distance_est":2.0,"lane_hint":"center"}],"free_space":{"left":0.75,"center":0.12,"right":0.30},"ego_state":{"speed_level":"slow"}}',
        '{"action":"BYPASS_LEFT","confidence":0.85,"reason":"center blocked at 12%, left 75% open"}'
    ),
    # 6. Center only 10%, right 65% → BYPASS_RIGHT
    (
        '{"obstacles":[{"type":"car","distance_est":1.8,"lane_hint":"center"},{"type":"bus","distance_est":2.5,"lane_hint":"center"}],"free_space":{"left":0.25,"center":0.10,"right":0.65},"ego_state":{"speed_level":"slow"}}',
        '{"action":"BYPASS_RIGHT","confidence":0.83,"reason":"center blocked at 10%, right 65% open"}'
    ),

    # ═══ SLOW_DOWN & WAIT ═══
    # 7. Center 35%, sides marginal → SLOW_DOWN
    (
        '{"obstacles":[{"type":"car","distance_est":2.0,"lane_hint":"center"},{"type":"car","distance_est":3.5,"lane_hint":"left"}],"free_space":{"left":0.60,"center":0.35,"right":0.55},"ego_state":{"speed_level":"medium"}}',
        '{"action":"SLOW_DOWN","confidence":0.70,"reason":"center 35% narrow, reduce speed"}'
    ),
    # 8. Center < 25%, both sides narrow, slow speed → WAIT
    (
        '{"obstacles":[{"type":"car","distance_est":1.5,"lane_hint":"center"}],"free_space":{"left":0.22,"center":0.10,"right":0.20},"ego_state":{"speed_level":"slow"}}',
        '{"action":"WAIT","confidence":0.75,"reason":"center blocked, both sides too narrow, wait"}'
    ),
]


# ── Model loading ────────────────────────────────────────
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float16 on MPS (fast) + NaN-safe fallback to greedy
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    print(f"  Model loaded on MPS GPU (float16, ~1.5s/frame, NaN-protected)")
    return tokenizer, model


def build_prompt(scene_json: str) -> list:
    """Build a few-shot chat prompt for TinyLlama Chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, asst_msg in FEW_SHOT:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": scene_json})
    return messages


def llm_decide(tokenizer, model, scene_json: str, debug=False) -> dict:
    """Send scene to TinyLlama, parse JSON output.
    Uses sampling (do_sample=True) with NaN-safe fallback."""
    messages = build_prompt(scene_json)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # ── Sampling params ──
    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ── Greedy fallback params ──
    greedy_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    result_text = ""
    used_fallback = False

    # Try sampling first
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        # MPS float16 NaN guard: check for NaN in output tokens
        if torch.isnan(outputs[0]).any():
            raise RuntimeError("NaN in output tokens (MPS float16 multinomial bug)")
        result_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not result_text or len(result_text) < 5:
            raise ValueError("Empty or too-short output from sampling")
    except Exception as e:
        if debug:
            print(f"[sampling error: {e}, falling back to greedy]", end=" ")
        used_fallback = True
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **greedy_kwargs)
            result_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e2:
            return {
                "action": "SLOW_DOWN", "confidence": 0.50,
                "reason": f"LLM error: {str(e2)[:40]}", "raw_output": ""
            }

    # Parse JSON
    try:
        start = result_text.find("{")
        end = result_text.rfind("}")
        if start >= 0 and end > start:
            json_str = result_text[start:end+1]
            parsed = json.loads(json_str)
            valid_actions = {"STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"}
            if parsed.get("action") not in valid_actions:
                parsed["action"] = "SLOW_DOWN"
            parsed["_fallback"] = used_fallback
            return parsed
    except json.JSONDecodeError:
        # If sampling produced bad JSON, retry with greedy
        if not used_fallback:
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **greedy_kwargs)
                result_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                start = result_text.find("{")
                end = result_text.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(result_text[start:end+1])
                    valid_actions = {"STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"}
                    if parsed.get("action") not in valid_actions:
                        parsed["action"] = "SLOW_DOWN"
                    parsed["_fallback"] = True
                    return parsed
            except:
                pass

    return {
        "action": "SLOW_DOWN", "confidence": 0.50,
        "reason": "LLM parse error", "raw_output": result_text[:80],
        "_fallback": True
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
    fallback_count = 0
    t0 = time.time()

    frames_with_obs = [r for r in results if r["scene_context"].get("obstacles")]
    print(f"\nProcessing {len(frames_with_obs)} frames with obstacles ({len(results)} total)")
    print(f"Sampling: do_sample=True, temp=0.6, top_p=0.92, top_k=50, MPS float16\n")

    for idx, result in enumerate(frames_with_obs):
        scene = result["scene_context"]
        decision_rule = result["decision"]

        # Build clean scene JSON
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

        center_val = scene["free_space"].get("center", 0)
        debug = (idx < 5)  # debug first 5 frames
        print(f"[{idx+1:3d}/{len(frames_with_obs)}] Frame {result['frame_id']:3d} center={center_val:.2f} …", end=" ", flush=True)
        t1 = time.time()
        decision_llm = llm_decide(tokenizer, model, scene_json, debug=debug)
        t_llm = (time.time() - t1) * 1000

        if decision_llm.get("_fallback"):
            fallback_count += 1

        action_llm = decision_llm.get("action", "SLOW_DOWN")
        confidence_llm = decision_llm.get("confidence", 0)
        action_rule = decision_rule.get("action", "N/A")

        match = "✅" if action_llm == action_rule else "⚠️"
        fb_mark = " [FB]" if decision_llm.get("_fallback") else ""
        print(f"{match} LLM:{action_llm}({confidence_llm:.2f}) vs Rule:{action_rule} [{t_llm:.0f}ms]{fb_mark}", flush=True)

        # Show raw output on mismatch for debugging
        if action_llm != action_rule and action_llm == "BYPASS_RIGHT":
            raw = decision_llm.get("raw_output", "")
            reason = decision_llm.get("reason", "")[:60]
            if reason:
                print(f"       └─ reason: {reason}", flush=True)

        actions_llm[action_llm] += 1
        actions_rule[action_rule] += 1

        decision_llm.pop("_fallback", None)
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
    print(f"Fallback (sampling→greedy retry): {fallback_count} frames")

    print(f"\nDecision distribution:")
    for act in ["STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"]:
        llm_cnt = actions_llm.get(act, 0)
        rule_cnt = actions_rule.get(act, 0)
        bar = "█" * min(llm_cnt, 60)
        print(f"  {act:14s} LLM={llm_cnt:3d}  Rule={rule_cnt:3d}  {bar}")

    # Show disagreements
    disagreements = [r for r in all_results if not r["match"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements[:15]:
            scene = r["scene"]
            obs_types = list(set(o["type"] for o in scene.get("obstacles", [])))
            free = scene.get("free_space", {})
            llm_act = r["decision_llm"]["action"]
            rule_act = r["decision_rule"]["action"]
            llm_reason = r["decision_llm"].get("reason", "")[:80]
            print(f"  Frame {r['frame_idx']:3d}: LLM={llm_act:14s} vs Rule={rule_act:14s} | center={free.get('center',0):.2f} | {llm_reason}")
        if len(disagreements) > 15:
            print(f"  ... and {len(disagreements)-15} more")

    # Save
    out_path = os.path.join(OUTPUT, "phase_a4_results_v3.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
