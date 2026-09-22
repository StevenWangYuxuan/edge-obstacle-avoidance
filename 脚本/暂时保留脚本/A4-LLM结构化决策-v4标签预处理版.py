#!/usr/bin/env python3
"""
Phase A4 v4: 数值→文字标签预处理 + 采样解码
核心思想: TinyLlama-1.1B 不会比数值大小，把它变成文字匹配问题
"""

from __future__ import annotations

import json, os, sys, time
from collections import Counter

import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SUMMARY  = os.path.expanduser("~/Documents/端测/输出/行车记录仪-YOLOv8n-FP32-GPU/phase_a1_board_summary.json")
OUTPUT   = os.path.expanduser("~/Documents/端测/输出/行车记录仪-LLM决策结果-v4")
os.makedirs(OUTPUT, exist_ok=True)

# ── 预处理：free_space 数值 → 文字标签 ─────────────────────
def label_free_space(val: float) -> str:
    """Convert free_space ratio to a text label the LLM can match."""
    if val >= 0.80:  return "WIDE"
    if val >= 0.45:  return "MODERATE"
    if val >= 0.25:  return "NARROW"
    return "BLOCKED"

def label_distance(d: float) -> str:
    """Classify obstacle distance for LLM."""
    if d < 0.8:   return "CRITICAL"
    if d < 2.0:   return "CLOSE"
    if d < 5.0:   return "MEDIUM"
    return "FAR"

# ImageNet 映射
IMAGENET_TO_DRIVING = {
    "moving van":"car","minivan":"car","limousine":"car","cab":"car",
    "police van":"car","pickup":"car","sports car":"car","racer":"car",
    "convertible":"car","beach wagon":"car","tow truck":"truck",
    "garbage truck":"truck","trailer truck":"truck","minibus":"bus",
    "passenger car":"car","amphibian":"car","reel":"car",
    "crane":"truck","snowplow":"truck","trolleybus":"bus","space shuttle":"truck",
    "motor scooter":"motorcycle","tricycle":"bicycle",
    "traffic light":"traffic light","stop sign":"traffic sign",
    "street sign":"traffic sign",
    "cassette player":"unknown","espresso maker":"unknown","milk can":"unknown",
    "lotion":"unknown","dishwasher":"unknown","oil filter":"unknown",
    "pill bottle":"unknown","binder":"unknown","cleaver":"unknown",
    "thresher":"unknown","book jacket":"unknown","chain saw":"unknown",
    "screen":"unknown","ski":"unknown","hand-held computer":"unknown",
    "desktop computer":"unknown","face powder":"unknown","Windsor tie":"unknown",
    "container ship":"unknown","bearskin":"unknown","packet":"unknown",
    "pole":"unknown","nipple":"unknown","beaker":"unknown",
    "rock beauty":"unknown","shoji":"unknown","slot":"unknown",
    "space bar":"unknown","guillotine":"unknown","cassette":"unknown",
    "affenpinscher":"unknown",
}

def map_type(label):
    if not label: return "unknown"
    if label in IMAGENET_TO_DRIVING: return IMAGENET_TO_DRIVING[label]
    low = label.lower()
    for k,v in IMAGENET_TO_DRIVING.items():
        if low == k.lower(): return v
    return label


# ── v4 Prompt: 仅用文字标签，无需数值比较 ──────────────────
SYSTEM_PROMPT = """You are a self-driving obstacle-avoidance controller.
Read the scene and output ONE action as JSON.

HOW TO READ THE INPUT:
- "road": where the car is heading — WIDE / MODERATE / NARROW / BLOCKED
- "left_side" / "right_side": side corridor width — WIDE / MODERATE / NARROW / BLOCKED
- "speed" and "obstacles" list with "distance" (FAR / MEDIUM / CLOSE / CRITICAL)

DECISION TABLE (exact match, first row wins):

| road      | condition                          | action        |
|-----------|------------------------------------|---------------|
| WIDE      | always                             | KEEP_CENTER   |
| MODERATE  | always                             | SLOW_DOWN     |
| NARROW    | left_side==WIDE or left_side==MODERATE | BYPASS_LEFT  |
| NARROW    | right_side==WIDE or right_side==MODERATE | BYPASS_RIGHT |
| NARROW    | speed==fast, no side open          | STOP          |
| NARROW    | speed!=fast, no side open          | WAIT          |
| BLOCKED   | left_side==WIDE                    | BYPASS_LEFT   |
| BLOCKED   | right_side==WIDE                   | BYPASS_RIGHT  |
| BLOCKED   | speed==fast                        | STOP          |
| BLOCKED   | speed!=fast                        | WAIT          |

SPECIAL RULE — obstacles within CRITICAL distance (less than 0.8m): output STOP
regardless of road width. Otherwise FAR obstacles should be ignored.

Output ONLY valid JSON, no other text:
{"action":"KEEP_CENTER","confidence":0.92,"reason":"road is WIDE, safe"}"""


# ── Few-shot: 全文字标签 ──────────────────────────────────
FEW_SHOT = [
    # 1. WIDE → KEEP_CENTER (最常见)
    (
        '{"road":"WIDE","left_side":"WIDE","right_side":"WIDE","speed":"slow","obstacles":[{"type":"car","distance":"FAR","position":"center"}]}',
        '{"action":"KEEP_CENTER","confidence":0.92,"reason":"road is WIDE, go straight"}'
    ),
    # 2. WIDE + 多障碍物 → 仍然是 KEEP_CENTER
    (
        '{"road":"WIDE","left_side":"MODERATE","right_side":"MODERATE","speed":"medium","obstacles":[{"type":"car","distance":"FAR","position":"left"},{"type":"truck","distance":"FAR","position":"right"}]}',
        '{"action":"KEEP_CENTER","confidence":0.90,"reason":"road WIDE, obstacles far"}'
    ),
    # 3. WIDE → KEEP_CENTER (场景像 KITTI: center=0.99 type of thing)
    (
        '{"road":"WIDE","left_side":"WIDE","right_side":"WIDE","speed":"slow","obstacles":[{"type":"car","distance":"FAR","position":"left"}]}',
        '{"action":"KEEP_CENTER","confidence":0.95,"reason":"road WIDE, just one far car"}'
    ),
    # 4. MODERATE → SLOW_DOWN
    (
        '{"road":"MODERATE","left_side":"WIDE","right_side":"NARROW","speed":"medium","obstacles":[{"type":"car","distance":"MEDIUM","position":"center"}]}',
        '{"action":"SLOW_DOWN","confidence":0.75,"reason":"road MODERATE, reduce speed"}'
    ),
    # 5. NARROW + left wide → BYPASS_LEFT
    (
        '{"road":"NARROW","left_side":"WIDE","right_side":"BLOCKED","speed":"slow","obstacles":[{"type":"truck","distance":"CLOSE","position":"center"}]}',
        '{"action":"BYPASS_LEFT","confidence":0.85,"reason":"road NARROW, left WIDE, bypass left"}'
    ),
    # 6. NARROW + right wide → BYPASS_RIGHT
    (
        '{"road":"NARROW","left_side":"BLOCKED","right_side":"WIDE","speed":"slow","obstacles":[{"type":"bus","distance":"CLOSE","position":"center"}]}',
        '{"action":"BYPASS_RIGHT","confidence":0.83,"reason":"road NARROW, right WIDE, bypass right"}'
    ),
    # 7. BLOCKED + left WIDE → BYPASS_LEFT
    (
        '{"road":"BLOCKED","left_side":"WIDE","right_side":"NARROW","speed":"slow","obstacles":[{"type":"car","distance":"CLOSE","position":"center"}]}',
        '{"action":"BYPASS_LEFT","confidence":0.88,"reason":"road BLOCKED, left WIDE, bypass left"}'
    ),
    # 8. BLOCKED + fast → STOP
    (
        '{"road":"BLOCKED","left_side":"BLOCKED","right_side":"BLOCKED","speed":"fast","obstacles":[{"type":"car","distance":"CLOSE","position":"center"}]}',
        '{"action":"STOP","confidence":0.96,"reason":"road BLOCKED, all sides blocked, speed fast, stop"}'
    ),
    # 9. CRITICAL distance → STOP (最高优先级)
    (
        '{"road":"WIDE","left_side":"WIDE","right_side":"WIDE","speed":"fast","obstacles":[{"type":"car","distance":"CRITICAL","position":"center"}]}',
        '{"action":"STOP","confidence":0.98,"reason":"CRITICAL obstacle at less than 0.8m, emergency stop"}'
    ),
    # 10. NARROW + no side → WAIT (slow)
    (
        '{"road":"NARROW","left_side":"NARROW","right_side":"BLOCKED","speed":"slow","obstacles":[{"type":"car","distance":"CLOSE","position":"center"}]}',
        '{"action":"WAIT","confidence":0.72,"reason":"road NARROW, no side open, speed slow, wait"}'
    ),
]


# ── 预处理: 把 raw scene 转成文字标签 JSON ──────────────────
def preprocess_scene(raw_scene: dict) -> dict:
    """Convert numerical free_space and distance to text labels."""
    free = raw_scene.get("free_space", {})
    ego  = raw_scene.get("ego_state", {})

    obstacles = []
    for o in raw_scene.get("obstacles", []):
        raw_type = o.get("yolo_label") or o.get("type", "unknown")
        obstacles.append({
            "type": map_type(raw_type),
            "distance": label_distance(o.get("distance_est", 8.0)),
            "position": o.get("lane_hint", "center"),
        })

    return {
        "road":       label_free_space(free.get("center", 1.0)),
        "left_side":  label_free_space(free.get("left", 1.0)),
        "right_side": label_free_space(free.get("right", 1.0)),
        "speed":      ego.get("speed_level", "slow"),
        "obstacles":  obstacles,
    }


# ── Model ─────────────────────────────────────────────────
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="mps", low_cpu_mem_usage=True)
    print(f"  MPS float16, sampling+NaN-guard")
    return tokenizer, model


def llm_decide(tokenizer, model, scene_json: str) -> dict:
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for u,a in FEW_SHOT:
        messages.append({"role":"user","content":u})
        messages.append({"role":"assistant","content":a})
    messages.append({"role":"user","content":scene_json})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Sampling + greedy fallback
    for attempt in range(2):
        use_sample = (attempt == 0)
        kwargs = dict(
            max_new_tokens=128, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=use_sample,
            temperature=0.6, top_p=0.92, top_k=50,
            repetition_penalty=1.15,
        ) if use_sample else dict(
            max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
            if torch.isnan(out[0]).any():
                raise RuntimeError("NaN in output")
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if not text or len(text) < 3:
                continue
            # Parse JSON
            s = text.find("{"); e = text.rfind("}")
            if s >= 0 and e > s:
                parsed = json.loads(text[s:e+1])
                if parsed.get("action") in {"STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"}:
                    return parsed
        except: continue

    return {"action":"KEEP_CENTER","confidence":0.50,"reason":"parse error","raw_output":""}


# ── Main ──────────────────────────────────────────────────
def main():
    with open(SUMMARY) as f:
        results = json.load(f)

    tokenizer, model = load_model()
    all_results = []
    actions_llm, actions_rule = Counter(), Counter()
    t0 = time.time()

    frames = [r for r in results if r["scene_context"].get("obstacles")]
    print(f"\nProcessing {len(frames)} frames | v4: text-label preprocessing | MPS float16\n")

    for idx, r in enumerate(frames):
        raw_scene = r["scene_context"]
        scene_labeled = preprocess_scene(raw_scene)
        scene_json = json.dumps(scene_labeled, ensure_ascii=False)
        rule_dec = r["decision"]

        print(f"[{idx+1:3d}/{len(frames)}] Frame {r['frame_id']:3d} road={scene_labeled['road']:8s} …", end=" ", flush=True)
        t1 = time.time()
        llm_dec = llm_decide(tokenizer, model, scene_json)
        dt = (time.time()-t1)*1000

        a_llm = llm_dec.get("action","?"); a_rule = rule_dec.get("action","?")
        match = "✅" if a_llm==a_rule else "⚠️"
        print(f"{match} LLM:{a_llm}({llm_dec.get('confidence',0):.2f}) vs Rule:{a_rule} [{dt:.0f}ms]", flush=True)
        if a_llm != a_rule:
            print(f"       └─ {llm_dec.get('reason','')[:80]}", flush=True)

        actions_llm[a_llm] += 1; actions_rule[a_rule] += 1
        all_results.append({
            "frame_idx": r["frame_id"], "scene_raw": raw_scene, "scene_labeled": scene_labeled,
            "decision_llm": llm_dec, "decision_rule": rule_dec,
            "llm_runtime_ms": round(dt), "match": a_llm==a_rule,
        })

    total_t = time.time()-t0; n=len(all_results)
    matches = sum(1 for x in all_results if x["match"])
    print(f"\n{'='*60}")
    print(f"v4 Decisions: {n} frames ({total_t:.1f}s, avg {total_t/n*1000:.0f}ms/frame)")
    print(f"Agreement: {matches}/{n} ({matches/n*100:.0f}%)")

    print(f"\nDecision distribution:")
    for act in ["STOP","WAIT","SLOW_DOWN","KEEP_CENTER","BYPASS_LEFT","BYPASS_RIGHT"]:
        lc = actions_llm.get(act,0); rc = actions_rule.get(act,0)
        bar = "█"*min(lc,60)
        print(f"  {act:14s} LLM={lc:3d}  Rule={rc:3d}  {bar}")

    # Show disagreements
    dis = [x for x in all_results if not x["match"]]
    print(f"\nDisagreements ({len(dis)}):")
    for x in dis[:10]:
        sl = x["scene_labeled"]
        print(f"  Frame {x['frame_idx']:3d}: LLM={x['decision_llm']['action']:14s} vs Rule={x['decision_rule']['action']:14s} | road={sl['road']} L={sl['left_side']} R={sl['right_side']} | obs={len(sl['obstacles'])}")
    if len(dis)>10: print(f"  ... and {len(dis)-10} more")

    out = os.path.join(OUTPUT, "phase_a4_results_v4.json")
    json.dump(all_results, open(out,"w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
