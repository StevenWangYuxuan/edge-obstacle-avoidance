# Day 10：Phase A 真机测试

> 日期：2026-07-03  
> 设备：QCS8550 / Kalama / Android 13  
> 目标：验证 Phase A 在真机上的第一版可执行链路

---

## 一、本次完成内容

本次已完成以下真机链路：

```text
KITTI 帧
-> 本机预处理为 YOLO 输入 raw
-> 真机 SNPE GPU 跑 YOLOv5s INT8
-> 回传 output0.raw
-> 本机后处理为 detections
-> 转 scene_context
-> 规则引擎输出动作
```

当前已经在真机上成功跑通 2 帧。

---

## 二、真机运行方式

### 1. 板端入口恢复

- 串口入口已恢复，可进入 Android shell
- `adbd` 在板上运行，监听 `5555`
- 真机当前 IP：`192.168.110.15`

### 2. 传输路径

由于 `adb devices` 仍处于 `offline` 状态，本次使用以下方式完成真机测试：

- Mac 本机启动临时 HTTP 服务
- 真机用 `curl` 拉取输入 raw
- 真机用 `snpe-net-run` 执行 YOLO
- 真机用 `nc` 把 `output0.raw` 回传到 Mac

这是可工作的替代路径，不阻塞 Phase A 推进。

---

## 三、真机 YOLO 结果

### 帧 0000000000

- 板端 YOLO 运行：
  - runtime: GPU
  - model: `yolov5s_quantized.dlc`
  - real time: `1.140 s`
- 检测结果：
  - `chair`
  - conf `0.943`
  - bbox `[417.9, 318.3, 497.1, 439.7]`
- scene_context 结果：
  - lane: `right`
  - free_space:
    - left `1.0`
    - center `1.0`
    - right `0.96`
- 决策结果：
  - action: `KEEP_CENTER`
  - confidence: `0.88`
  - reason: `center corridor is clear`

### 帧 0000000001

- 板端 YOLO 运行：
  - runtime: GPU
  - model: `yolov5s_quantized.dlc`
  - real time: `1.021 s`
- 检测结果：
  - `chair`
  - conf `0.905`
  - bbox `[429.8, 319.3, 515.2, 451.2]`
- scene_context 结果：
  - lane: `right`
  - free_space:
    - left `1.0`
    - center `1.0`
    - right `0.95`
- 决策结果：
  - action: `KEEP_CENTER`
  - confidence: `0.88`
  - reason: `center corridor is clear`

---

## 四、当前结论

本次已经证明：

1. Phase A 主链可以真正落到真机执行
2. 真机 YOLO 输出与 VM 侧解析结果一致
3. `detections -> scene_context -> 决策动作` 这条链已具备真实输入支撑

也就是说，Phase A 已经不是纯本地原型，而是进入了“真机可运行”的状态。

---

## 五、当前未完成项

本次尚未完成：

1. 真机板端 InceptionV3 分类补充链
2. 多帧批量真机自动化
3. 全链路都在板上本地完成，不再依赖 Mac 做后处理
4. `adb offline` 问题彻底修复

---

## 六、建议下一步

优先顺序建议：

1. 把本次 2 帧扩展到 10 帧真机批量测试
2. 接上真机 InceptionV3 crop 分类
3. 将 `detections_to_scene.py` 与 `realtime_decision.py` 迁到板上直接运行
4. 最后再收口 `adb offline`
