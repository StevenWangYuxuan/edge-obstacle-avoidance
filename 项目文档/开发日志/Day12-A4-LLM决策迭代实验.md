# Day 12：A4-LLM 结构化决策四轮迭代实验

> 日期：2026-07-06
> 结论：TinyLlama-1.1B 无法可靠做结构化决策，数值→文字标签约束解码是下一步方向

---

## 一、实验背景

Day 11 的 Phase A 中，TinyLlama-1.1B 在 KITTI 全 KEEP_CENTER 数据上显示 100% 一致（v1）。但此后数据中出现了非 KEEP_CENTER 的场景，需要验证 LLM 在更复杂的场景分布下是否仍能与规则引擎保持一致。

---

## 二、四轮实验记录

### 实验环境

| 项目 | 配置 |
|------|------|
| 设备 | Mac Apple Silicon |
| 模型 | TinyLlama-1.1B-Chat-v1.0 |
| 推理后端 | MPS GPU（Apple Silicon） |
| 测试数据 | 173 帧 KITTI 行车记录仪，166 帧有障碍物 |
| 规则引擎 | 6 动作：STOP / WAIT / SLOW_DOWN / KEEP_CENTER / BYPASS_LEFT / BYPASS_RIGHT |
| 数据分布 | center>0.80: 164帧, 0.45-0.80: 8帧, <0.25: 1帧 |

### v1 — 原始版（参考基准）

| 属性 | 值 |
|------|-----|
| 策略 | 贪心解码, MPS float16 |
| 一致率 | **13%**（旧数据上显示 100%，但因数据全是 KEEP_CENTER，是巧合而非能力） |
| KEEP_CENTER | 0/118 |
| BYPASS_RIGHT | 132/23 |
| 核心问题 | 模型把 free_space=0.99 误解为"0.99米距离"而非"99%开放比例" |

```
脚本：A4-LLM结构化决策.py
模型加载：device_map="auto", float16
解码：do_sample=False (greedy)
```

### v2 — 优化版（CPU + 决策树 prompt）

| 属性 | 值 |
|------|-----|
| 策略 | 贪心解码, CPU float32, 决策树式 prompt + 10 个 few-shot |
| 一致率 | **13%**（与 v1 相同） |
| 速度 | CPU ~12s/帧，MPS GPU ~1.4s/帧 |
| 结论 | 贪心解码下 prompt 优化无效 — 模型死循环 BYPASS_RIGHT |

```
脚本：A4-LLM结构化决策-v2优化版.py
模型加载：CPU float32 (后改为 MPS float16)
解码：do_sample=False (greedy)
新增：ImageNet→驾驶语义标签映射, 决策树式 prompt
```

### v3 — 采样优化版（恢复 do_sample + NaN 保护）

| 属性 | 值 |
|------|-----|
| 策略 | do_sample=True, temp=0.6, float16 MPS + NaN→greedy 回退 |
| 一致率 | **22%** — 提升但远不够 |
| 速度 | ~2.1s/帧（回退开销） |
| NaN 回退率 | 39%（float16 在 MPS 上 multinomial NaN 频繁） |
| BYPASS_RIGHT | **0** ✅ 成功打破死循环 |
| KEEP_CENTER | 40 ✅ 从零到有 |
| WAIT | 78 ⚠️ 新问题：过度谨慎，同一 0.95 场景有时 KEEP_CENTER 有时 WAIT |

```
脚本：A4-LLM结构化决策-v3采样优化版.py
模型加载：MPS float16
解码：do_sample=True, temp=0.6, top_p=0.92, top_k=50
NaN保护：torch.isnan检测→回退贪心解码
Prompt：解释 free_space 是比例, 强化 KEEP_CENTER 规则, 8 个 few-shot
```

### v4 — 文字标签预处理版

| 属性 | 值 |
|------|-----|
| 策略 | 把 free_space 数值→WIDE/MODERATE/NARROW/BLOCKED 文字标签 |
| 一致率 | **19%** — 反而下降 |
| 速度 | ~1.7s/帧 |
| KEEP_CENTER | 39 |
| WAIT | 119 ⚠️ 仍然过度等待 |
| BYPASS_RIGHT | 0 |
| 核心发现 | LLM 的 reason **读懂了标签**（"road WIDE, obstacles FAR"），**但决策表匹配失败** — pretraining bias 压倒 prompt 指令 |

```
脚本：A4-LLM结构化决策-v4标签预处理版.py
模型加载：MPS float16
预处理：free_space 0.0-1.0 → WIDE(≥0.80) / MODERATE(≥0.45) / NARROW(≥0.25) / BLOCKED(<0.25)
        distance_est → FAR / MEDIUM / CLOSE / CRITICAL
解码：do_sample=True + NaN保护 + greedy回退
Prompt：全文字标签的决策表
```

---

## 三、四轮对比总结

| 版本 | 策略 | 一致率 | KEEP_CENTER | BYPASS_RIGHT | 每帧速度 | 结论 |
|------|------|--------|-------------|-------------|---------|------|
| v1 | 贪心+数值 | 13% | 0 | 132 | ~1.4s | 死循环 BYPASS_RIGHT |
| v2 | 贪心+决策树+数值 | 13% | 0 | 132 | 1.4-12s | prompt 优化对贪心解码无效 |
| v3 | 采样+NaN保护+数值 | **22%** | 40 | 0 | 2.1s | 打破死循环但 WAIT 过度 |
| v4 | 采样+文字标签 | 19% | 39 | 0 | 1.7s | 标签读懂，决策表匹配失败 |

### 所有版本的共性问题

```
reason 摘录：
  "center blocked at 0.99"           ← 模型把 0.99 当米，不是 99%
  "wait for traffic lights"          ← 场景里根本没有红绿灯，幻觉
  "road WIDE, wait"                  ← 知道路宽却说等
  "road WIDE, all sides blocked"     ← 明明 sides 也是 WIDE，幻觉
```

---

## 四、模型诊断

### 为什么 TinyLlama-1.1B 不适合做结构化 if-then 决策？

| 原因 | 解释 |
|------|------|
| **1.1B 参数不够** | 指令遵循能力弱，pretraining bias 压倒 prompt |
| **数值比较能力缺失** | 语言模型的核心是用 pattern 预测下一个 token，不是做数值比较 |
| **Chat 微调 bias** | 作为 Chat 模型，它被训练得更"安全、谨慎、会拒绝"——在有障碍物时天然倾向 WAIT/STOP |
| **幻觉频繁** | 凭空生成"traffic lights"、"speed FAST"、"sides blocked" |

### MPS GPU 上 TinyLlama 的 NaN bug

- `do_sample=True` + `float16` 在 MPS 上 multinomial 采样会出 NaN
- `float32` 稳定但速度慢（~7s/帧）
- 最佳方案：`float16` + NaN 检测 + 贪心 fallback

---

## 五、QNN 部署现实

在上板之前，LLM 必须在 Mac/VM 上完成 QNN 转换 + 量化。但当前工具链存在多个断裂点：

| 步骤 | 状态 |
|------|------|
| HuggingFace → ONNX | ✅ 可行 |
| ONNX → DLC | ⚠️ TinyLlama(~6GB ONNX) 远超 VM 可处理范围 |
| INT8 量化 | ❌ DistilGPT-2(82M) 都 segfault，TinyLlama(1.1B) 必崩 |
| INT4 量化 | ❌ 同上 |
| 板端 HTP 推理 | ❌ SDK v2.0.1 ≠ v2.22.6 |
| 板端 CPU 推理 | ✅ ONNX Runtime Android 可用，但 >10s/帧 |

---

## 六、下一步方向

### 短期可做

1. **约束解码（Constrained Decoding）**：文字标签预处理 → 规则引擎给出安全候选集 → LLM 在候选内选优 + 生成理由（已记录为 future direction）
2. **换 Qwen2.5-1.5B-Instruct**：在 Mac 上跑 1.5B 比 TinyLlama-1.1B 大 36%，指令遵循能力强得多

### 中期需解决

3. **获取匹配版 QNN SDK** + ARM64 CLI 工具 → 打通 ONNX→DLC→HTP 全链路
4. **在板端部署 ONNX Runtime** 作为过渡方案（CPU 推理 2-5s/帧）

### 长期

5. **LoRA 微调**：用规则引擎输出做 ground truth，微调 TinyLlama / Qwen 使其学会数值决策
6. **QNN v2.28+ LLM native support**：新版 SDK 原生支持 Llama 架构量化

---

## 七、产出文件

| 文件 | 路径 |
|------|------|
| v1 脚本 | `脚本/暂时保留脚本/A4-LLM结构化决策.py` |
| v2 脚本 | `脚本/暂时保留脚本/A4-LLM结构化决策-v2优化版.py` |
| v3 脚本 | `脚本/暂时保留脚本/A4-LLM结构化决策-v3采样优化版.py` |
| v4 脚本 | `脚本/暂时保留脚本/A4-LLM结构化决策-v4标签预处理版.py` |
| 可视化脚本 | `脚本/暂时保留脚本/A4-生成LLM对比视频.py` |
| v2 结果 | `输出/行车记录仪-LLM决策结果-v2/phase_a4_results_v2.json` |
| v3 结果 | `输出/行车记录仪-LLM决策结果-v3/phase_a4_results_v3.json` |
| v4 结果 | `输出/行车记录仪-LLM决策结果-v4/phase_a4_results_v4.json` |
| v2 视频 | `输出/LLM对比可视化-行车记录仪/phase_a4_comparison.mp4` |
| v3 视频 | `输出/LLM对比可视化-行车记录仪-v3/phase_a4_comparison.mp4` |
| v4 视频 | `输出/LLM对比可视化-行车记录仪-v4/phase_a4_comparison.mp4` |

---

## 八、经验教训

1. **不要用 1.1B 模型做数值 if-then**：这是语言模型、不是计算器
2. **贪心解码 vs 采样的 tradeoff**：贪心→死循环；采样→多样性但 NaN 频繁
3. **prompt 优化有天花板**：再好的 prompt，1.1B 也学不会数值比较
4. **MPS float16 NaN bug**：do_sample=True 时必须加 NaN 保护
5. **VM 不是 LLM QNN 的可行环境**：TinyLlama ONNX ~6GB，VM 上 ONNX→DLC 至少数小时/可能 OOM
