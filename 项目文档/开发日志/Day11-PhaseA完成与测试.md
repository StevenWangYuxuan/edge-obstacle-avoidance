# Day 11：Phase A 完成 + 三轮板端测试 + 项目整理

> 日期：2026-07-03  
> 目标：完成 Phase A 全四任务，三轮 KITTI 数据板端测试，项目文件整理

---

## 一、Phase A 总览

| 任务 | 内容 | 结果 |
|------|------|------|
| A1 | YOLO + InceptionV3 板端管线 | ✅ 三轮共 243 帧跑通 |
| A2 | 规则引擎 | ✅ 6 决策路径 |
| A3 | JSON 协议 | ✅ scene_context + decision_output |
| A4 | TinyLlama 结构化决策 | ✅ 与规则引擎一致 100%/94%/100% |

---

## 二、测试记录

### 2.1 第一轮：KITTI drive_0048（高速，28 帧）
- 模型：INT8
- 后端：CPU
- 决策：KEEP_CENTER × 28
- 可视化：`输出/视频1-高速28帧-可视化/`

### 2.2 第二轮：KITTI drive_0052（住宅区，84 帧）
- 模型：INT8
- 后端：CPU
- 决策：KEEP_CENTER × 84
- LLM 一致率：94% (32/34)
- 可视化：`输出/视频2-住宅区84帧-可视化/`

### 2.3 第三轮：KITTI drive_0046（城市道路，131 帧）
- 模型：FP32
- 后端：CPU
- 决策：KEEP_CENTER × 131
- LLM 一致率：100% (24/24)
- 可视化：`输出/城市道路131帧-可视化/`

### 2.4 合成障碍物测试
- 方案：OpenCV 几何图形 + 真车贴图
- 结果：❌ YOLO FP32/INT8 均不识别合成物体

### 2.5 GPU 对比测试
- GPU 逐帧 ~1500ms > CPU ~900ms（每帧重建 GPU 上下文）
- GPU 持续流可达 ~39ms（Day 8 已验证）
- 结论：逐帧测试用 CPU，持续视频流才用 GPU

---

## 三、LLM 调试历程

> ⚠️ **2026-07-06 更正**：Day 11 的 LLM"100%/94%/100%一致"结论不准确。当时使用的是 YOLOv5 数据，其 3350 次 traffic light 误检导致 free_space 计算失真→规则引擎 173 帧全出 KEEP_CENTER→LLM 只需学会说 KEEP_CENTER 即可"100%一致"——这是数据巧合，不是模型能力。
>
> 2026-07-06 用 YOLOv8n 的准确数据重测后，TinyLlama-1.1B 在真正的多动作场景下（166 帧含 STOP/BYPASS/SLOW_DOWN/KEEP_CENTER）一致率仅 **13-22%**。详见 Day 12 完整四轮实验记录。

| 版本 | 修复 | 测试数据 | 真实一致率 | 说明 |
|------|------|---------|-----------|------|
| v1 | 原始 prompt | YOLOv5(全KEEP_CENTER) | — | 不可参考 |
| v2 | 加决策矩阵 + KITTI example | YOLOv5(全KEEP_CENTER) | — | 不可参考 |
| v3 | 硬规则 `center>0.80 → KEEP_CENTER` | YOLOv5(全KEEP_CENTER) | — | 不可参考 |
| **v3 重测** | 同上 | **YOLOv8(5种action)** | **22%** | KEEP_CENTER: 0→40 |
| **v4 重测** | 文字标签预处理 | **YOLOv8(5种action)** | **19%** | BYPASS_RIGHT: 0 ✅ |

- 模型：TinyLlama-1.1B-Chat-v1.0
- 运行位置：Mac M5 Pro (MPS GPU)，~1.6s/帧
- 无法在板端运行：HTP SDK 版本不匹配

---

## 四、项目整理

重新组织为中文命名结构：

```
📁 端测/
├── 📁 项目文档/    ← 计划/日报/参考资料
├── 📁 脚本/        ← 6 个核心脚本 + LLM工具链旧目录
├── 📁 测试数据/    ← 3 批 KITTI + 合成障碍物
├── 📁 输出/        ← 推理JSON + 可视化视频/GIF
├── 📁 协议定义/    ← 2 个 JSON schema
└── 📁 工具/        ← 串口通信
```

---

## 五、关键结论

1. **视觉检测链路完整可用**：YOLO → InceptionV3 → 场景 JSON → 规则引擎，在 QCS8550 板端跑通
2. **LLM 决策逻辑验证通过**：但只能在 Mac 上跑，等 HTP 版本匹配后搬上板
3. **KITTI 不适合此项目**：三批 243 帧全是畅通场景，需换测试数据
4. **HTP 是唯一卡点**：板端 SDK v2.0.1 vs 手头 v2.22.6，需联系板厂解决
