# Day 6：LLM 端侧部署 — ONNX 导出与 QNN 转换

> 完成日期：2026-07-01
> 模型：DistilGPT-2 (81.9M 参数, 6 层 Transformer)
> 工具链：optimum-cli → ONNX → qairt-converter → QNN IR
> 目标环境：QCS8550 (骁龙 8 Gen 2) · HTP · LPAI backend

---

## 一、目标

将一个小型 GPT 模型通过 QNN 工具链转换为 DLC 格式，证明 LLM → QNN 这条路线是通的，为后续 INT4 量化和管线串联打基础。

### 为什么选 DistilGPT-2

| 模型 | 参数量 | FP32 大小 | 预估 INT4 | 理由 |
|------|--------|-----------|-----------|------|
| DistilGPT-2 | 81.9M | ~328 MB | ~82 MB | 最小可用 GPT，HTP 能跑得动 |
| TinyLlama-1.1B | 1.1B | ~4.4 GB | ~1.1 GB | 理想方案，但需更大内存 |
| Qwen2-1.5B | 1.5B | ~6 GB | ~1.5 GB | 中文好，但偏大 |

> 从最小的 DistilGPT-2 开始验证管线，跑通后再换更大的模型。

---

## 二、完成清单

- [x] QNN LPAI backend 环境确认（QNN SDK v2.22.6）
- [x] `optimum-cli export onnx` 导出 DistilGPT-2
- [x] 修复 ONNX 图中 QNN 不支持的 `IsNaN` 算子（6个）
- [x] `qairt-converter` 转换 ONNX → QNN IR
- [ ] `qairt-quantizer` INT4 量化（待 Day 7）
- [ ] QNN CPU/HTP 推理验证（待 Day 7）

---

## 三、关键步骤

### 3.1 环境确认 ✅

```bash
# QNN SDK 提供的 LLM 相关能力
qnn-onnx-converter --help
# --apply_masked_softmax     → Transformer attention 优化
# --packed_masked_softmax_inputs → 多序列打包
# --pack_4_bit_weights       → INT4 权重

# LPAI backend 支持的目标
# QNN_LPAI_BACKEND_TARGET_X86   = 0
# QNN_LPAI_BACKEND_TARGET_ARM   = 1  
# QNN_LPAI_BACKEND_TARGET_ADSP  = 2
```

### 3.2 ONNX 导出 ✅

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -m optimum.exporters.onnx \
  --model distilgpt2 \
  --task text-generation \
  --opset 18 \
  --batch_size 1 \
  --sequence_length 32 \
  01_models/distilgpt2_onnx_v18/
```

**产出：** `model.onnx` (460MB, 1496 节点, 30 种算子)

### 3.3 ONNX 图修复 ✅

QNN converter v2.22.6 不支持 `IsNaN` 算子。利用 IEEE 754 标准中 `NaN ≠ NaN` 的特性：

```
IsNaN(x)  ≡  Not(Equal(x, x))
```

为 6 个 attention head 各替换一个 `Equal` 节点，并调整下游 `Where` 的条件极性。

**脚本：** `/Users/steven/Documents/端测/scripts/remove_isnan.py`

**产出：** `distilgpt2_qnn.onnx` (460MB, IsNaN: 0)

### 3.4 QNN 转换 ✅

```bash
qairt-converter \
  --input_network 01_models/distilgpt2_qnn.onnx \
  -d input_ids 1,32 \
  -d attention_mask 1,32 \
  -d position_ids 1,32 \
  --output_path 01_models/distilgpt2_seq32.bin \
  --float_bitwidth 32
```

**产出：** `distilgpt2_seq32.bin` (461MB, QNN IR 格式)

**WARNING（已知，不影响）：**
- `WARNING_GEMM: GEMM → FC` — 全连接层自动重映射
- `WARNING_OP_VERSION_NOT_SUPPORTED: Split v18 → v13` — 自动降级

---

## 四、工作目录结构

```
~/day6-llm/
├── 01_models/
│   ├── distilgpt2_onnx/              ← opset 14 (有问题, 保留备份)
│   ├── distilgpt2_onnx_v18/          ← opset 18 原始导出 (460MB)
│   ├── distilgpt2_sim.onnx           ← onnxsim 简化后
│   ├── distilgpt2_qnn.onnx           ← IsNaN 替换后 (460MB)
│   └── distilgpt2_seq32.bin          ← QNN IR 转换结果 (461MB) 🎯
├── 02_scripts/
│   ├── export_gpt2_onnx.py           ← 手动 export (失败)
│   └── export_gpt2_onnx_v2.py        ← 手动 export v2 (失败)
└── 03_output/
```

---

## 五、踩坑记录

| # | 描述 | 解法 |
|----|------|------|
| 16 | PyTorch 2.12.1 `torch.onnx.export` 强制依赖 onnxscript，多个版本组合均失败 | 改用 `optimum-cli export onnx` |
| 17 | optimum 触发的 `protobuf ≥ 4.25` 与 TF 2.10.1 不兼容 | `pip install protobuf==3.20.3` |
| 18 | opset 13 不支持 `aten::scaled_dot_product_attention` | 升到 opset 18 |
| 19 | ONNX 导出含 `IsNaN` 算子（6个），QNN converter 不支持 | Python 脚本用 `Equal(x,x)` 等价替换 + 调整 Where 极性 |

---

## 六、当前链路进度

```
视频输入 → YOLOv5s (✅ Day 4) → InceptionV3 (✅ Day 3) → LLM DistilGPT-2
                                                               │
                                                  ┌────────────┘
                                                  │
                                          ONNX 导出 (✅ Day 6)
                                                  │
                                          QNN IR 转换 (✅ Day 6)
                                                  │
                                          INT4 量化 (🔲 Day 7)
                                                  │
                                          CPU/HTP 推理 (🔲 Day 7)
```

### 对应 QCS8550 技术栈

```
DistilGPT-2 (HuggingFace)
  → optimum-cli → ONNX (opset 18)
  → remove_isnan → 修复不兼容算子
  → qairt-converter → QNN IR (.bin)
  → qairt-quantizer → INT4 DLC (待做)
  → qnn-net-run → HTP/CPU 推理 (待做)
      ↓
  QCS8550 Hexagon HTP @ 48 TOPS
```

---

## 七、下一步：Day 7

- `qairt-quantizer` INT4 量化 distilgpt2_seq32.bin
- QNN CPU 推理验证（生成一段文本）
- 测量 LLM 推理延迟和内存
- 对比量化前后模型大小
