#!/usr/bin/env python3
"""
Phase A4: TinyLlama-1.1B structured decision brain
Input: scene_context JSON → Output: structured action JSON
Uses few-shot prompting with 8-action space.
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
OUTPUT = os.path.expanduser("~/Documents/端测/输出/行车记录仪-LLM决策结果")
os.makedirs(OUTPUT, exist_ok=True)

# ── System prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are a self-driving car obstacle-avoidance controller.
You receive a scene JSON and must output ONE action from the 8 options below.

CRITICAL RULES (follow strictly, in order):

HARD RULE: If center free_space > 0.80, you MUST output KEEP_CENTER.
No exceptions. No matter how many obstacles exist. No matter their types.
0.80 center = wide open path = KEEP_CENTER. Period.

ONLY output STOP if: center < 0.25 AND nearest obstacle distance < 1.0m.
ONLY output BYPASS if: center < 0.25 AND one side has free_space > 0.40.
ONLY output WAIT if: center < 0.25 AND both sides < 0.25 AND speed is "slow".
Otherwise output KEEP_CENTER.

distance_est=8.0 means extremely far away. Ignore such obstacles.

Output EXACTLY: {"action":"X","confidence":0.XX,"reason":"..."}
No other text. Actions: STOP WAIT SLOW_DOWN KEEP_CENTER BYPASS_LEFT BYPASS_RIGHT"""

# ── Few-shot examples ───────────────────────────────────
FEW_SHOT = [
    # Example 1: Center blocked, left clear → BYPASS_LEFT
    (
        '{"obstacles":[{"type":"cone","distance_est":1.2,"lane_hint":"center"}],"free_space":{"left":0.82,"center":0.18,"right":0.34},"ego_state":{"speed_level":"slow","heading":"forward"}}',
        '{"action":"BYPASS_LEFT","confidence":0.84,"reason":"center blocked, left corridor wider than right"}'
    ),
    # Example 2: REAL emergency → STOP (only when center < 0.25 AND distance < 1m)
    (
        '{"obstacles":[{"type":"car","distance_est":0.6,"lane_hint":"center"},{"type":"cone","distance_est":0.5,"lane_hint":"center"}],"free_space":{"left":0.15,"center":0.05,"right":0.10},"ego_state":{"speed_level":"fast","heading":"forward"}}',
        '{"action":"STOP","confidence":0.95,"reason":"emergency: center 0.05 blocked, nearest obstacle 0.5m"}'
    ),
    # Example 3: Highway — cars in side lanes, center wide open → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"left"},{"type":"truck","distance_est":8.0,"lane_hint":"right"}],"free_space":{"left":0.55,"center":0.95,"right":0.50},"ego_state":{"speed_level":"fast","heading":"forward"}}',
        '{"action":"KEEP_CENTER","confidence":0.92,"reason":"center corridor wide open at 0.95, obstacles are far away in side lanes"}'
    ),
    # Example 4: KITTI-like — one car ahead but far, center clear → KEEP_CENTER
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"center"}],"free_space":{"left":1.0,"center":0.97,"right":1.0},"ego_state":{"speed_level":"slow","heading":"forward"}}',
        '{"action":"KEEP_CENTER","confidence":0.90,"reason":"center free_space 0.97 is wide open, obstacle is far at 8.0m distance"}'
    ),
    # Example 4b: TWO obstacles but center STILL clear → KEEP_CENTER (this is important!)
    (
        '{"obstacles":[{"type":"car","distance_est":8.0,"lane_hint":"left"},{"type":"truck","distance_est":8.0,"lane_hint":"left"}],"free_space":{"left":0.85,"center":0.95,"right":1.0},"ego_state":{"speed_level":"medium","heading":"forward"}}',
        '{"action":"KEEP_CENTER","confidence":0.90,"reason":"center free_space 0.95 is wide open, obstacles far away in left lane only"}'
    ),
    # Example 5: Ambiguous center → SLOW_DOWN
    (
        '{"obstacles":[{"type":"car","distance_est":1.8,"lane_hint":"center"}],"free_space":{"left":0.35,"center":0.30,"right":0.38},"ego_state":{"speed_level":"medium","heading":"forward"}}',
        '{"action":"SLOW_DOWN","confidence":0.62,"reason":"center partially blocked, both sides marginal"}'
    ),
    # Example 6: No obstacles → KEEP_CENTER
    (
        '{"obstacles":[],"free_space":{"left":0.90,"center":0.92,"right":0.88},"ego_state":{"speed_level":"fast","heading":"forward"}}',
        '{"action":"KEEP_CENTER","confidence":0.92,"reason":"no obstacles, all corridors clear"}'
    ),
    # Example 7: Right blocked, left open → BYPASS_LEFT
    (
        '{"obstacles":[{"type":"cone","distance_est":0.9,"lane_hint":"center"},{"type":"barrier","distance_est":1.1,"lane_hint":"right"}],"free_space":{"left":0.70,"center":0.15,"right":0.20},"ego_state":{"speed_level":"slow","heading":"forward"}}',
        '{"action":"BYPASS_LEFT","confidence":0.82,"reason":"center blocked, left corridor has most free space"}'
    ),
]

# ── Model loading ───────────────────────────────────────
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"  Model loaded. Device: {model.device}")
    return tokenizer, model

def build_prompt(scene_json: str) -> str:
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
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.05,
            do_sample=False,  # greedy: avoids MPS multinomial NaN bug
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    result_text = result_text.strip()

    # Try to extract JSON from output
    try:
        # Find JSON boundaries
        start = result_text.find("{")
        end = result_text.rfind("}")
        if start >= 0 and end > start:
            json_str = result_text[start:end+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fallback: return raw text
    return {"action": "SLOW_DOWN", "confidence": 0.50, "reason": "LLM parse error", "raw_output": result_text[:100]}

# ── Main ─────────────────────────────────────────────────
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

    # Only process frames with obstacles (skip empty frames to save time)
    frames_with_obs = [r for r in results if r["scene_context"].get("obstacles")]
    print(f"\nProcessing {len(frames_with_obs)} frames with obstacles ({len(results)} total)\n")

    for idx, result in enumerate(frames_with_obs):
        scene = result["scene_context"]
        decision_rule = result["decision"]

        # Build clean scene JSON for LLM
        scene_clean = {
            "obstacles": [
                {"type": o["type"], "distance_est": o["distance_est"], "lane_hint": o["lane_hint"]}
                for o in scene.get("obstacles", [])
            ],
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
        reason_llm = decision_llm.get("reason", "")

        action_rule = decision_rule.get("action", "N/A")

        # Comparison marker
        match = "✅" if action_llm == action_rule else "⚠️"
        print(f"{match} LLM: {action_llm}({confidence_llm:.2f}) vs Rule: {action_rule} [{t_llm:.0f}ms]")

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

    # ── Summary ───────────────────────────────────────────
    n = len(all_results)
    matches = sum(1 for r in all_results if r["match"])
    print(f"LLM decisions: {n} frames ({total_t:.1f}s, avg {total_t/n*1000:.0f}ms/frame)")
    print(f"Agreement with rule engine: {matches}/{n} ({matches/n*100:.0f}%)")
    print(f"\nLLM decisions:")
    for act, cnt in actions_llm.most_common():
        rule_cnt = actions_rule.get(act, 0)
        print(f"  {act}: LLM={cnt}, Rule={rule_cnt}")

    # Show disagreements
    disagreements = [r for r in all_results if not r["match"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements:
            scene = r["scene"]
            obs_types = [o["type"] for o in scene.get("obstacles", [])]
            print(f"  Frame {r['frame_idx']}: LLM={r['decision_llm']['action']} vs Rule={r['decision_rule']['action']} | obstacles={obs_types}")

    # Save
    out_path = os.path.join(OUTPUT, "phase_a4_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
