# Day 9：双层决策实施起步

> 日期：2026-07-03  
> 目标：把“LLM 产品化替代路线”从设计推进到第一版可运行原型

---

## 一、本次新增产物

### 1. 协议

- `schemas/scene_context.schema.json`
  - 定义视觉层输出给决策层的结构化场景输入
- `schemas/decision_output.schema.json`
  - 定义规则层和后续 LLM 层统一输出的动作格式

### 2. 样例数据

- `examples/sample_scene.json`
  - 一帧障碍场景的示例输入
- `examples/sample_detections.json`
  - 一帧来自 YOLO/分类层的示例检测结果

### 3. 实时层原型

- `scripts/realtime_decision.py`
  - 第一版确定性规则引擎
  - 输入结构化场景 JSON
  - 输出固定动作集合中的一个动作
- `scripts/detections_to_scene.py`
  - 视觉层桥接脚本
  - 输入检测结果 JSON
  - 输出 `scene_context`

---

## 二、当前实现边界

当前规则引擎已经能处理：

- 正前方近距离障碍的紧急停车
- 中路被挡时的左右绕行选择
- 两侧都不安全时的等待/减速
- 场景不明确时的保守回退

当前还没有实现：

- 直接读取 `output0.raw` 做自动后处理
- 多帧时序平滑
- LLM 层融合
- 距离估计模型
- 可通行区域分割/深度估计

---

## 三、下一步接法

### 1. 视觉层接入

把 YOLO + Inception 的输出整理成：

```json
{
  "frame_id": 123,
  "obstacles": [...],
  "free_space": {...},
  "ego_state": {...}
}
```

当前已具备第一版桥接能力：

```bash
python3 scripts/detections_to_scene.py examples/sample_detections.json
```

### 2. 规则层先闭环

先让 `realtime_decision.py` 单独跑通：

```bash
python3 scripts/realtime_decision.py examples/sample_scene.json
```

现在也可以直接串起来：

```bash
python3 scripts/detections_to_scene.py examples/sample_detections.json > /tmp/scene.json
python3 scripts/realtime_decision.py /tmp/scene.json
```

### 3. LLM 层后接入

后续让 LLM 只消费同一个 `scene_context`，输出同一个 `decision_output`。

这样实时层和 LLM 层天然可以做融合，不需要两套协议。

---

## 四、为什么这一步重要

这一步把项目从：

```text
概念：YOLO -> 分类 -> LLM -> 动作
```

推进到了：

```text
协议已定义
+ 实时规则层可运行
+ 后续 LLM 有明确接入边界
```

这意味着后面即使 HTP 还没恢复，系统主链也已经可以继续长。
