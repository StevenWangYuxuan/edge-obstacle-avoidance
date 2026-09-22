"""
Export DistilGPT-2 to ONNX for QNN LPAI backend verification.
Smallest GPT-2 variant (82M params) with fixed input shape.
"""
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Loading DistilGPT-2...")
from transformers import GPT2LMHeadModel, AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
model.config.use_cache = False

batch_size = 1
seq_len = 32
input_shape = (batch_size, seq_len)

dummy_input = torch.randint(0, 100, input_shape, dtype=torch.long)
attention_mask = torch.ones(input_shape, dtype=torch.long)

print(f"Input shape: {input_shape}")
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

output_path = os.path.expanduser("~/day6-llm/01_models/distilgpt2_seq32.onnx")

with torch.no_grad():
    output = model(dummy_input, attention_mask=attention_mask)
    print(f"Output shape: {output.logits.shape}")

print(f"Exporting to ONNX: {output_path}")
torch.onnx.export(
    model,
    (dummy_input, attention_mask),
    output_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=13,
    do_constant_folding=True,
)

import onnx
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX model verified: {output_path}")

size_mb = os.path.getsize(output_path) / 1e6
print(f"ONNX file size: {size_mb:.1f} MB")
print("Done!")
