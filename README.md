# 端侧障碍规避系统 — 高通 QCS8550 全链路部署

> 课题：基于端侧 AI，输入视频流 → 识别路障 → 自主路线规划 → 规避障碍
> 目标硬件：高通 QCS8550（骁龙 8 Gen 2 / HTP V73，48 TOPS，24GB RAM，Android 13）
> 工具链：Qualcomm SNPE/QNN SDK v2.22.6

将「视频流 → 障碍检测 → 细粒度分类 → 结构化场景建模 → 规避决策」的视觉感知与决策链路在高通 QCS8550 板端实现并验证。针对通用 LLM 直接进入实时闭环导致的时延抖动与输出不可控问题，**设计了**双层决策架构：确定性规则引擎承担实时决策，小参数 LLM 作为非实时策略顾问。

> ⚠️ **当前实现边界（请先读）**
> 板端实际跑通的是**视觉链路**（YOLO + InceptionV3 + 规则引擎）。LLM 因板端 HTP 不可用，**仅在 Mac MPS 上离线评估**，未上板，也未与规则引擎接入同一融合层。端到端管线实测约 **3.5 秒/帧，未达实时要求**。详见「核心结果」与「已知限制」。

---

## 一、系统架构

```
                        视频流 (30fps)
                              │
                    逐帧提取 + 前处理
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        ┌──────────────┐            ┌──────────────┐
        │   YOLOv8n    │            │ InceptionV3  │
        │  障碍检测     │            │  障碍细分类   │
        └──────┬───────┘            └──────┬───────┘
               └─────────────┬─────────────┘
                             ▼
                  结构化场景描述 (scene_context)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │   实时决策层     │   ◄───   │  LLM 高层策略层  │
     │   规则引擎       │          │  结构化小脑      │
     │   毫秒级、确定性  │          │  非实时顾问      │
     └────────┬────────┘          └────────┬────────┘
              └──────────────┬─────────────┘
                             ▼
                     融合决策 → 规避动作
```

**核心设计原则：实时层不依赖 LLM 也能独立工作。** LLM 缺失 / 超时 / 低置信度时自动回退到规则动作。该约束使项目在板端 HTP 工具链全线阻塞的情况下，仍能完成整个 Phase A 的系统验证。

> 上图为**设计目标**。实际实现状态：视觉链路（YOLO + InceptionV3）与规则引擎已在板端跑通，LLM 层仅离线评估，融合层未实现——即图中「LLM 高层策略层 → 融合决策」这一段目前**只存在于设计中**。

**动作空间固定为 8 个**（可评测、可量化、便于约束解码）：
`STOP` / `WAIT` / `SLOW_DOWN` / `KEEP_CENTER` / `TURN_LEFT` / `TURN_RIGHT` / `BYPASS_LEFT` / `BYPASS_RIGHT`

协议以 JSON Schema 固化（见 `协议定义/`）：
- `scene_context` — `obstacles[]`（归一化坐标 / 距离估计 / 车道归属 / 置信度）+ `free_space{left,center,right}` + `ego_state`
- `decision_output` — `{action, confidence, reason, source}`

---

## 二、核心结果

### 2.1 端侧推理性能（QCS8550 真机实测）

性能数据分三个层级，含义不同，**请勿混用**：

**① 模型吞吐** —— `snpe-throughput-net-run`，10 秒连续批量，用总时长除以帧数得到的**摊销值**

| 模型 | 后端 | 吞吐 | 摊销单帧 |
|------|------|------|---------|
| YOLOv5s FP32 | SNPE GPU | 25.83 inf/s | 38.7 ms |
| InceptionV3 FP32 | SNPE GPU | 32.53 inf/s | 30.7 ms |
| YOLOv5s FP32 | SNPE CPU | 3.91 inf/s | 255.9 ms |
| InceptionV3 FP32 | SNPE CPU | 10.48 inf/s | 95.4 ms |

> 该指标只反映模型在循环中连续推理的速度，**不含前处理、后处理、adb 往返与文件 I/O**，不代表管线延迟。

**② 单次端到端** —— `snpe-net-run`，含模型加载 + 图构建 + 一次推理 + 输出写入

| 模型 | 后端 | real |
|------|------|------|
| YOLOv5s FP32 | SNPE GPU | 1.016 s |
| InceptionV3 FP32 | SNPE GPU | 0.664 s |

**③ 实际管线逐帧耗时**（173 帧行车记录仪数据）

| 环节 | 耗时 |
|------|------|
| YOLOv8n 推理 | 715 ms |
| InceptionV3（逐检测框裁剪分类） | ~2700 ms |
| **端到端合计** | **~3467 ms（约 3.5 s/帧）** |

**结论：管线未达到实时要求。** 瓶颈不在模型本身——InceptionV3 的 GPU 吞吐可达 32.5 inf/s——而在它需要对**每个检测框逐个裁剪后再分类**，检测框一多耗时即线性增长。优化方向见「已知限制」。

**LLM（未上板）**：TinyLlama-1.1B 在 Mac MPS 上 1.4–2.1 s/帧；板端 CPU >10 s/帧，无实用价值；板端 HTP 因版本不匹配不可用。

量化压缩：InceptionV3 92MB → 24MB（**-74%**）；YOLOv5s 29MB → 7.2MB（**-75%**）。

> 注：x86 CPU 上 INT8 并不加速（反而慢 2.7%），因为缺少 INT8 向量指令、需运行时反量化。板端 GPU 上 INT8 与 FP32 吞吐亦几乎持平（38.7 vs 38.7 ms）。INT8 的收益必须依赖 DSP/HTP 硬件。

### 2.2 检测模型迁移：YOLOv5s → YOLOv8n（同一 173 帧数据集）

| 指标 | YOLOv5s | YOLOv8n | 变化 |
|------|---------|---------|------|
| 板端推理耗时 | 1653ms | **715ms** | **2.3x 加速** |
| 检测框总数 | 4325 | **778** | **-82%** |
| traffic light 误检 | 3350 | 142 | **-96%** |
| 平均置信度 | 0.936（过自信） | 0.436 | 更真实 |
| **下游规则决策多样性** | 1 种 | **5 种** | 质变 |

**连锁失效归因**：YOLOv5 的 3350 次 traffic light 误检挤占了可通行区域（free_space）估算 → 每帧 `center < 0.25` → 下游规则引擎 173 帧全部退化为同一个动作。迁移至 YOLOv8n 后 free_space 真实反映路况，决策多样性由 1 种恢复至 5 种，才使后续决策层实验具备有效场景分布。

### 2.3 LLM 结构化决策实验（四轮受控迭代）

| 版本 | 策略 | 一致率 | 主要症状 |
|------|------|--------|----------|
| v1 | 贪心解码 + 数值 prompt | 13% | 死循环 `BYPASS_RIGHT`（132/173 帧） |
| v2 | 贪心解码 + 决策树 prompt + Few-shot | 13% | prompt 优化对贪心解码无效 |
| v3 | 采样解码 + NaN 保护 | **22%** | 打破死循环，但过度 `WAIT`（78 帧） |
| v4 | 采样 + 数值→文字标签预处理 | 19% | 标签读懂，决策表匹配失败 |

**结论**：TinyLlama-1.1B 不具备 if-then 数值比较能力 —— 将 `free_space=0.99` 误解为「0.99 米距离」，且 Chat 微调带来的谨慎偏置压倒 prompt 指令。这是模型能力天花板，而非 prompt 工程问题。据此提出 **约束解码** 改进方向：规则引擎不直接输出动作，而是产出安全候选集，LLM 仅在候选集内选优并生成理由（**尚未实现**）。

> 实验形式说明：这是**离线评估** —— LLM 读取已生成好的 `scene_context` JSON 独立决策，再与规则引擎的输出比对一致率。LLM **未接入实时闭环**，融合层（Phase B4）尚未实现。

---

## 三、目录结构

```
├── 脚本/
│   ├── A1-板端推理管线-yolov8.py      # 主推理管线（Mac 编排 → 板端 SNPE GPU）
│   ├── A1-板端推理管线.py             # YOLOv5s 版本（保留作对比基线）
│   ├── 规则引擎.py                    # A2 实时决策层，确定性 fallback controller
│   ├── 检测结果转场景.py               # 检测框 → scene_context
│   ├── A1-生成可视化视频.py            # 检测结果可视化
│   ├── 生成障碍物测试数据.py
│   └── 暂时保留脚本/
│       ├── A4-LLM结构化决策*.py        # v1–v4 四轮实验脚本
│       ├── A4-生成LLM对比视频.py
│       └── LLM工具链-旧/               # ONNX 导出与算子修复（DistilGPT-2 验证）
├── 协议定义/                          # A3 产出：JSON Schema 双协议
├── 项目文档/
│   ├── 开发日志/Day1–Day12.md          # 逐日开发记录与踩坑
│   ├── 项目计划/                       # 完整计划 / 进度报告 / 架构论证
│   └── 参考资料/imagenet_slim_labels.txt
├── 示例数据/                          # scene_context / detections 样例
└── 工具/serial_cmd.py                 # 板端串口调试
```

---

## 四、快速开始

### 4.1 环境依赖

```bash
# 开发机（Mac）：负责编排与结果解析
pip install numpy opencv-python

# 板端：SNPE 运行时 + adb 连接
adb devices                     # 确认板子已连接
```

### 4.2 模型准备

仓库不含模型权重。原始 `yolov8n.pt` 从 Ultralytics 官方获取后自行转换：

```bash
# PyTorch → ONNX
yolo export model=yolov8n.pt format=onnx opset=18

# ONNX → DLC（需 Qualcomm SNPE SDK，建议 ARM64 环境下执行）
snpe-onnx-to-dlc -i yolov8n.onnx -o yolov8n.dlc

# 推送至板端
adb push yolov8n.dlc        /data/local/tmp/models/
adb push inception_v3.dlc   /data/local/tmp/models/
```

### 4.3 运行

```bash
# 全链路：帧 → YOLOv8n → InceptionV3 → scene_context → decision
adb shell "echo OK" && python3 脚本/A1-板端推理管线-yolov8.py

# 单帧决策验证（只需 scene JSON）
python3 脚本/规则引擎.py 示例数据/sample_scene.json
```

### 4.4 测试数据

- **KITTI** 需从 [官方站点](https://www.cvlibs.net/datasets/kitti/) 自行申请下载（CC BY-NC-SA，非商用）。本项目使用 drive_0046 / 0048 / 0052 共 243 帧。
- **行车记录仪** 173 帧为自采数据，未随仓库分发。

---

## 五、关键技术点与踩坑

| 类别 | 问题 | 解法 |
|------|------|------|
| SNPE | ONNX opset 18 不兼容 | 降 opset + 属性修复，转 DLC 固定 onnx 1.13 / protobuf 3.20 |
| SNPE | YOLOv8 输出为单张量 `[1,84,8400]`，坐标是绝对像素 | 重写解码逻辑，无独立 objectness 通道 |
| QNN | LLM 量化 5 次 segfault（82M Transformer / 461MB / 1496 节点） | 排除内存、校准数据、DLC 损坏后定性为 SDK 工具链缺陷 —— 见「已知限制」 |
| QNN | ONNX 含 `IsNaN` 算子，QNN 不支持 | Python 脚本以 `Equal(x,x)` 等价替换 |
| QNN | 动态 `Gather` 索引导致校验失败 | 以 `Constant[index]` 替换为静态索引 |
| PyTorch | Apple MPS float16 下 multinomial 采样产生 NaN | NaN 检测 + 贪心解码回退（回退率 39%） |
| 部署 | 板端 HTP runtime `v2.0.1` ≠ SDK `v2.22.6` | 报 `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES`，转向 SNPE GPU 后端 |

完整 21 条踩坑记录见 `项目文档/项目计划/端测项目总进度报告-2026-07-02.md`。

---

## 六、已知限制与后续方向

**当前限制**

- **未达实时**：端到端管线约 3.5 s/帧（瓶颈为 InceptionV3 的逐检测框裁剪分类），与 30fps 需求差距明显。
- **LLM 未上板**：板端 HTP 因 vendor runtime `v2.0.1.220706112757` 与 SDK `v2.22.6` 版本不匹配而全线不可用；LLM 仅在 Mac MPS 上离线评估，未在板端运行过。
- **双层架构未集成**：规则引擎与 LLM 各自独立验证，融合层（Phase B4）尚未实现。
- **决策层能力不足**：1.1B 级模型不具备结构化数值决策能力，四轮实验已证伪。
- **测试场景单一**：KITTI 243 帧几乎全为畅通场景，需更丰富的真实障碍物数据。

**后续方向**

1. **约束解码** —— 数值预处理为文字标签，规则引擎产出安全候选集，LLM 在候选集内选优 + 生成理由（安全边界由规则保证，泛化由 LLM 承担，且预处理极轻量）。
2. 换用 **Qwen2.5-1.5B-Instruct** 复测指令遵循能力。
3. 获取匹配版 QNN SDK / 新版 BSP，打通 HTP 全链路。
4. 以规则引擎输出为 ground truth 做 **LoRA 微调**。

---

## 七、文档索引

| 文档 | 内容 |
|------|------|
| `项目文档/项目计划/端侧障碍规避系统-完整计划.md` | 系统架构、技术栈、Phase A/B/C 路线图 |
| `项目文档/项目计划/端侧障碍规避系统-LLM产品化替代路线.md` | 双层决策架构的完整论证 |
| `项目文档/项目计划/端测项目总进度报告-2026-07-02.md` | 总进度、性能数据、21 条踩坑汇总 |
| `项目文档/开发日志/Day12-A4-LLM决策迭代实验.md` | LLM 四轮实验完整记录与归因 |
| `项目文档/开发日志/Day8-QCS8550真机性能测试.md` | 板端真机性能实测 |
