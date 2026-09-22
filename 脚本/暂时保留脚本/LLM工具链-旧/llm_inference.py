#!/usr/bin/env python3
"""
LLM FP32 DLC Inference Pipeline
Tokenize → raw → qnn-net-run → decode next token
"""
import numpy as np
import os
import subprocess
import sys
import struct

SNPE_RUN = os.path.expanduser("/home/steven/qairt/2.22.6.240515/bin/x86_64-linux-clang/snpe-net-run")
DLC_PATH = os.path.expanduser("/home/steven/day6-llm/01_models/dlc/distilgpt2_fp32.dlc")
OUTPUT_DIR = os.path.expanduser("/home/steven/day6-llm/03_output/inference")
TMP_DIR = os.path.expanduser("/home/steven/day6-llm/03_output/tmp")
SEQ_LEN = 32
VOCAB_SIZE = 50257

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── Step 1: Tokenize ─────────────────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
# Use local files from the ONNX export directory (contains full tokenizer)
model_dir = os.path.expanduser("/home/steven/day6-llm/01_models/distilgpt2_onnx_v18")
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "The capital of France is",
    "The quick brown fox jumps over the",
    "Artificial intelligence will transform",
]

for pi, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Prompt [{pi}]: \"{prompt}\"")

    tokens = tokenizer.encode(prompt)
    print(f"Token count: {len(tokens)}")

    # Pad/truncate to SEQ_LEN
    if len(tokens) > SEQ_LEN:
        tokens = tokens[:SEQ_LEN]
    pad_len = SEQ_LEN - len(tokens)
    input_ids = tokens + [tokenizer.eos_token_id] * pad_len
    attention_mask = [1] * len(tokens) + [0] * pad_len
    position_ids = list(range(SEQ_LEN))

    # Convert to numpy float32
    ids_np = np.array(input_ids, dtype=np.float32).reshape(1, SEQ_LEN)
    mask_np = np.array(attention_mask, dtype=np.float32).reshape(1, SEQ_LEN)
    pos_np = np.array(position_ids, dtype=np.float32).reshape(1, SEQ_LEN)

    # Write raw files
    ids_path = os.path.join(TMP_DIR, f"input_ids_{pi}.raw")
    mask_path = os.path.join(TMP_DIR, f"attention_mask_{pi}.raw")
    pos_path = os.path.join(TMP_DIR, f"position_ids_{pi}.raw")
    ids_np.tofile(ids_path)
    mask_np.tofile(mask_path)
    pos_np.tofile(pos_path)

    # Write input list
    list_path = os.path.join(TMP_DIR, f"input_list_{pi}.txt")
    with open(list_path, "w") as f:
        f.write(f"{ids_path} {mask_path} {pos_path}\n")

    # ── Step 2: QNN Inference ─────────────────────────
    out_dir = os.path.join(OUTPUT_DIR, f"prompt_{pi}")
    print(f"Running qnn-net-run...")

    cmd = [
        SNPE_RUN,
        "--container", DLC_PATH,
        "--input_list", list_path,
        "--output_dir", out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
        continue

    print(f"Inference complete")

    # ── Step 3: Parse output ───────────────────────────
    # QNN output naming: Result_X/<tensor_name>.raw
    result_dirs = [d for d in os.listdir(out_dir) if d.startswith("Result_")]
    if not result_dirs:
        print(f"No result dirs in {out_dir}")
        print(subprocess.run(["ls", "-R", out_dir], capture_output=True, text=True).stdout)
        continue

    raw_file = None
    for rd in result_dirs:
        rdir = os.path.join(out_dir, rd)
        for f in os.listdir(rdir):
            if f.endswith(".raw"):
                raw_file = os.path.join(rdir, f)
                break

    if not raw_file:
        print("No output .raw file found")
        continue

    # Read as float32, reshape to (1, SEQ_LEN, VOCAB_SIZE)
    data = np.fromfile(raw_file, dtype=np.float32)
    print(f"Output size: {data.shape[0]} floats, expected {SEQ_LEN * VOCAB_SIZE}")

    if data.shape[0] >= SEQ_LEN * VOCAB_SIZE:
        logits = data[:SEQ_LEN * VOCAB_SIZE].reshape(SEQ_LEN, VOCAB_SIZE)
    else:
        print(f"Unexpected output size: {data.shape}")
        continue

    # ── Step 4: Decode ─────────────────────────────────
    # Get next-token prediction (last valid token position)
    valid_len = len(tokens)
    next_logits = logits[valid_len - 1]  # logits at last prompt token
    top5_idx = np.argsort(next_logits)[-5:][::-1]
    top5_probs = np.exp(next_logits[top5_idx]) / np.sum(np.exp(next_logits[top5_idx]))

    predicted_token = top5_idx[0]
    predicted_text = tokenizer.decode([predicted_token])

    print(f"\n┌──────────────────────────────────────┐")
    print(f"│ Next token prediction               │")
    print(f"├──────────────────────────────────────┤")
    for i, (tid, prob) in enumerate(zip(top5_idx, top5_probs)):
        marker = "←" if i == 0 else "  "
        tok_text = tokenizer.decode([tid]).replace("\n", "\\n")
        print(f"│ {marker} {repr(tok_text):20s}  ({prob:.3f}) │")
    print(f"└──────────────────────────────────────┘")
    print(f"\n→ Completion: \"{prompt} {predicted_text}\"")

print("\n\nDone!")
