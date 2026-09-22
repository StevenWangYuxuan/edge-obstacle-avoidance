# Day 8：QCS8550 真机端侧性能测试

> 测试日期：2026-07-02  
> 设备：Orion O8G2 / Qualcomm QCS8550 (SM8550, Snapdragon 8 Gen 2)  
> OS：Android 13, Linux 5.15.78, Kalama  
> SNPE：v2.22.6.240515184619_92920  
> 连接方式：Mac 串口 `/dev/cu.usbserial-A10LCBPJ` @ 115200  
> 测试路径：`/data/local/tmp`

---

## 一、结论摘要

本轮已在 QCS8550 真机上跑通 YOLOv5s 和 InceptionV3 的 FP32 / INT8 四组模型。

当前有效性能数据来自 **SNPE CPU runtime** 和 **SNPE GPU runtime**。DSP/HTP 硬件存在但运行时验证失败，暂未拿到 HTP 推理数据。

| 模型 | 版本 | Runtime | 吞吐 (inf/s) | 单帧耗时 (ms, 由吞吐换算) | 单次端到端 real time |
|------|------|---------|--------------|----------------------------|----------------------|
| YOLOv5s | FP32 | CPU | 3.7684 | 265.4 ms | 328 ms |
| YOLOv5s | INT8 | CPU | 3.9071 | 255.9 ms | 345 ms |
| InceptionV3 | FP32 | CPU | 10.4820 | 95.4 ms | 207 ms |
| InceptionV3 | INT8 | CPU | 10.3997 | 96.2 ms | 226 ms |
| YOLOv5s | FP32 | GPU | 25.8284 | 38.7 ms | 1016 ms |
| YOLOv5s | INT8 | GPU | 25.8098 | 38.7 ms | 1015 ms |
| InceptionV3 | FP32 | GPU | 32.5264 | 30.7 ms | 664 ms |
| InceptionV3 | INT8 | GPU | 32.5330 | 30.7 ms | 690 ms |

关键观察：

- QCS8550 CPU 比之前 VM x86 模拟环境快很多：YOLO 从约 7 秒/帧提升到约 0.26 秒/帧。
- GPU 持续吞吐提升明显：YOLO GPU 约 25.8 inf/s，接近 39 ms/帧；Inception GPU 约 32.5 inf/s，约 31 ms/张。
- CPU runtime 下 INT8 没有明显加速：YOLO INT8 仅比 FP32 快约 3.7%，Inception INT8 反而略慢约 0.8%。
- GPU runtime 显示为 `gpu_float32_16_hybrid`，FP32/INT8 吞吐几乎相同，说明 GPU 并没有走 INT8 专用整数加速路径。
- INT8 的模型体积压缩明显，约 75%；真正推理加速仍需 HTP/DSP 跑通。

---

## 二、设备与运行环境

串口进入 Android shell 后确认：

```text
Linux localhost 5.15.78-android13-8-g94468358cb85-ab46 #1 SMP PREEMPT Thu Jun 27 02:20:26 UTC 2024 aarch64 Toybox
uid=2000(shell) gid=2000(shell) groups=2000(shell),1007(log),3009(readproc) context=u:r:shell:s0
ro.product.model = Kalama for arm64
ro.board.platform = kalama
ro.build.version.release = 13
```

SNPE 工具：

```text
/data/local/tmp/snpe-net-run
/data/local/tmp/snpe-throughput-net-run
/data/local/tmp/qnn-net-run
SNPE v2.22.6.240515184619_92920
```

GPU 验证：

```text
Backend GPU Prerequisites: Present.
Library Version of the backend GPU: OpenCL 3.0 Adreno(TM) 740
Core Version of the backend GPU: Adreno(TM) 740
Unit Test on the backend GPU: Passed.
QNN is supported for backend GPU on the device.
```

---

## 三、模型文件

| 模型 | 文件 | 大小 |
|------|------|------|
| YOLOv5s FP32 | `/data/local/tmp/models/yolov5s.dlc` | 28 MB |
| YOLOv5s INT8 | `/data/local/tmp/models/yolov5s_quantized.dlc` | 7.1 MB |
| InceptionV3 FP32 | `/data/local/tmp/models/inception_v3.dlc` | 91 MB |
| InceptionV3 INT8 | `/data/local/tmp/models/inception_v3_int8.dlc` | 23 MB |

压缩效果：

| 模型 | FP32 | INT8 | 体积下降 |
|------|------|------|----------|
| YOLOv5s | 28 MB | 7.1 MB | 约 74.6% |
| InceptionV3 | 91 MB | 23 MB | 约 74.7% |

---

## 四、吞吐测试

测试命令模板：

```bash
export LD_LIBRARY_PATH=/data/local/tmp/qnn_libs:/vendor/lib64:$LD_LIBRARY_PATH

./snpe-throughput-net-run \
  --container <model.dlc> \
  --duration 10 \
  --use_cpu \
  --perf_profile burst \
  --input_raw <input.raw> \
  --json <report.json>
```

### 4.1 YOLOv5s FP32

```text
runtime: cpu_float32
num_images: 38
duration: 10084115 us
throughput: 3.768399707414 inf/s
```

换算单帧耗时：约 265.4 ms。

### 4.2 YOLOv5s INT8

```text
runtime: cpu_float32
num_images: 40
duration: 10238083 us
throughput: 3.907059264912 inf/s
```

换算单帧耗时：约 255.9 ms。

### 4.3 InceptionV3 FP32

```text
runtime: cpu_float32
num_images: 105
duration: 10017480 us
throughput: 10.482014958734 inf/s
```

换算单帧耗时：约 95.4 ms。

### 4.4 InceptionV3 INT8

```text
runtime: cpu_float32
num_images: 104
duration: 10000540 us
throughput: 10.399718167638 inf/s
```

换算单帧耗时：约 96.2 ms。

### 4.5 YOLOv5s FP32 GPU

```text
runtime: gpu_float32_16_hybrid
num_images: 259
duration: 10028477 us
throughput: 25.828439724898 inf/s
```

换算单帧耗时：约 38.7 ms。

### 4.6 YOLOv5s INT8 GPU

```text
runtime: gpu_float32_16_hybrid
num_images: 259
duration: 10035763 us
throughput: 25.809766475425 inf/s
```

换算单帧耗时：约 38.7 ms。

### 4.7 InceptionV3 FP32 GPU

```text
runtime: gpu_float32_16_hybrid
num_images: 326
duration: 10023494 us
throughput: 32.526441452655 inf/s
```

换算单帧耗时：约 30.7 ms。

### 4.8 InceptionV3 INT8 GPU

```text
runtime: gpu_float32_16_hybrid
num_images: 326
duration: 10021406 us
throughput: 32.533043742673 inf/s
```

换算单帧耗时：约 30.7 ms。

---

## 五、单次端到端测试

测试命令模板：

```bash
toybox time ./snpe-net-run \
  --container <model.dlc> \
  --input_list <input_list.txt> \
  --output_dir <output_dir> \
  --perf_profile burst \
  --profiling_level basic
```

| 模型 | 版本 | real | user | sys |
|------|------|------|------|-----|
| YOLOv5s | FP32 | 0.328 s | 0.824581 s | 0.726716 s |
| YOLOv5s | INT8 | 0.345 s | 0.762649 s | 0.829559 s |
| InceptionV3 | FP32 | 0.207 s | 0.439332 s | 0.289470 s |
| InceptionV3 | INT8 | 0.226 s | 0.462870 s | 0.283283 s |
| YOLOv5s GPU | FP32 | 1.016 s | 0.826135 s | 0.128116 s |
| YOLOv5s GPU | INT8 | 1.015 s | 0.808764 s | 0.144391 s |
| InceptionV3 GPU | FP32 | 0.664 s | 0.513459 s | 0.102459 s |
| InceptionV3 GPU | INT8 | 0.690 s | 0.525496 s | 0.116975 s |

端到端时间包含模型加载、图构建、一次推理和输出文件写入，所以通常比吞吐换算出的稳定单帧耗时更高。

GPU 单次端到端 real time 明显高于 CPU，是因为首次图构建和 GPU 初始化开销较大；持续吞吐测试更能反映视频流/批量推理场景的 GPU 价值。

---

## 六、DSP/HTP 状态

QCS8550 的 DSP/HTP 硬件可识别：

```text
Core Version of the backend DSP: Hexagon Architecture V73
Backend Hardware  : Supported
Backend Libraries : Found
```

但 QNN/SNPE DSP 后端验证未通过：

```text
Unit Test on the backend DSP: Failed.
QNN is NOT supported for backend DSP on the device.
```

SNPE 平台验证器也显示：

```text
Runtime DSP Prerequisites: Absent.
SNPE is NOT supported for runtime DSP on the device.
Core Version of the runtime DSP: Hexagon Architecture V73
```

一次 YOLOv5s INT8 `--use_dsp` 冒烟测试失败：

```text
QnnBackend_DeviceCreate() failed;
QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES: Loaded libraries are of incompatible versions
```

当前判断：

- 板子硬件是 V73 HTP，没有问题。
- `/data/local/tmp/qnn_libs` 里的 SDK runtime 与板上 DSP skeleton / FastRPC 环境可能存在版本或签名不匹配。
- 需要修复 HTP 环境后再测 INT8 DSP/HTP，届时才是最终真实性能。

### 6.1 HTP 修复排查进展

已进一步确认版本不匹配：

```text
/vendor/lib64/libQnnHtp.so       -> v2.0.1.220706112757_38277
/data/local/tmp/qnn_libs/*       -> SNPE/QNN v2.22.6.240515
```

这解释了 `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES`：当前测试工具和 `/data/local/tmp/qnn_libs` 来自 QNN 2.22.6，但板端 vendor HTP runtime/skel 是 2.0.1 时代的 BSP 组件。

也确认 SDK 2.22.6 里有 V73 unsigned skel：

```text
htp_fix_assets/libQnnHtpV73Skel.so
htp_fix_assets/libCalculator_skel.so
```

文件校验：

```text
bf2d46f8db909495ed780300fe5b58e04d698782983971e5da99205632c67dbe  libQnnHtpV73Skel.so
7b129f35100066bc4e522adecd9efbbb8c81c6916ca38ee4045fe18e5c1257b2  libCalculator_skel.so
```

但加载 unsigned DSP 镜像通常需要打开 FastRPC unsigned/testsig 调试路径。该操作会降低设备安全边界，本轮未自动执行。

建议修复路线：

1. 优先路线：拿到与板端 BSP 匹配的 QNN/SNPE 2.0.1 工具和 aarch64 runtime，避免混用 2.22.6 runtime。
2. 正式路线：升级板端 `/vendor/lib64/libQnn*` 和 `/vendor/lib/rfsa/adsp/libQnnHtpV73Skel.so` 到与 QNN 2.22.6 匹配的 BSP/runtime。
3. 开发板调试路线：在明确允许的前提下，启用 FastRPC unsigned/testsig，临时加载 SDK 2.22.6 的 unsigned V73 skel 做验证。

### 6.2 方案 1 执行状态

已在本地、VM、板端 BSP 中查找匹配 QNN/SNPE 2.0.1 工具链：

```text
本地工作区：未找到 2.0.1 SDK/runner
VM：仅有 /home/steven/qairt/2.22.6.240515
板端 /vendor,/system,/odm,/product：仅有 runtime 库，没有 qnn-net-run/snpe-net-run
Qualcomm Software Center 旧版直链：可跳转到 QSC API，但返回 403，需要账号/权限或供应商包
```

板端现有 vendor runtime：

```text
/vendor/lib64/libQnnCpu.so
/vendor/lib64/libQnnHtp.so
/vendor/lib64/libQnnHtpPrepare.so
/vendor/lib64/libQnnHtpV73Stub.so
/vendor/lib64/libQnnSystem.so
/vendor/lib/rfsa/adsp/libQnnHtpV73Skel.so
```

需要向板厂/供应商索取的最小包：

```text
QNN/SNPE SDK or runtime package matching:
  QNN HTP version: v2.0.1.220706112757_38277
  Target: Android aarch64
  SoC: SM8550 / QCS8550 / Kalama
  HTP arch: V73

Required files:
  bin/aarch64-android/qnn-net-run
  bin/aarch64-android/qnn-throughput-net-run
  bin/aarch64-android/qnn-platform-validator
  bin/aarch64-android/snpe-net-run                 # if using SNPE DLC path
  bin/aarch64-android/snpe-throughput-net-run      # if using SNPE DLC path
  lib/aarch64-android/libQnnModelDlc.so            # if using DLC directly with QNN
  Matching aarch64 runtime libs, if not relying on /vendor/lib64
```

拿到包后验证顺序：

```bash
qnn-platform-validator --backend dsp --coreVersion --libVersion --testBackend
snpe-platform-validator --runtime dsp --coreVersion
snpe-throughput-net-run --container models/yolov5s_quantized.dlc --duration 10 --use_dsp --input_raw test_input.raw
```

---

## 七、和 Day 5 VM 数据对比

| 模型 | VM x86 CPU | QCS8550 CPU | 提升 |
|------|------------|-------------|------|
| YOLOv5s FP32 | 0.16 inf/s | 3.77 inf/s | 约 23.6x |
| YOLOv5s INT8 | 0.15 inf/s | 3.91 inf/s | 约 26.0x |
| InceptionV3 FP32 | 0.30 inf/s | 10.48 inf/s | 约 34.9x |

| 模型 | QCS8550 CPU | QCS8550 GPU | GPU/CPU |
|------|-------------|-------------|---------|
| YOLOv5s FP32 | 3.77 inf/s | 25.83 inf/s | 约 6.9x |
| YOLOv5s INT8 | 3.91 inf/s | 25.81 inf/s | 约 6.6x |
| InceptionV3 FP32 | 10.48 inf/s | 32.53 inf/s | 约 3.1x |
| InceptionV3 INT8 | 10.40 inf/s | 32.53 inf/s | 约 3.1x |

说明：

- VM 是 x86-on-ARM 虚拟环境，性能数据只能验证流程，不代表真实端侧。
- QCS8550 即使只跑 CPU runtime，也已经远快于 VM。
- GPU 已经可以把 YOLO 推到接近 25 fps；如果只看模型推理本身，接近实时。
- INT8 的核心价值仍在 HTP/DSP；CPU 下主要体现模型体积压缩。

---

## 七-b、YOLOv5s → YOLOv8n 升级 (2026-07-06 补充)

因 YOLOv5s 对 KITTI 画面大量误检（3350 次 traffic light 误检/173 帧→free_space 失真→173 帧全判 KEEP_CENTER），升级为 YOLOv8n DLC 后重测。

### 性能对比 (SNPE GPU, 173 帧 KITTI)

| 指标 | YOLOv5s DLC | YOLOv8n DLC | 变化 |
|------|------------|------------|------|
| GPU 推理 (单帧) | 1653ms (含上下文重建) | **715ms** | **2.3x 快** |
| 管道总耗时 | 3478ms | 3467ms | ≈ |
| 总检测数 | 4325 | 778 | **减少 82%** |
| 平均检测/帧 | 25.0 | 4.5 | 精准化 |
| traffic light 误检 | 3350 | 142 | **消除 96%** |
| 规则决策多样性 | 1 种 | **5 种** | 质的提升 |

### 总耗时未降原因

瓶颈转移：YOLO 从 1653→715ms (-938ms)，但 InceptionV3 + adb 推拉 + 场景构建约 2700ms 未变。

---

## 八、下一步

1. 修复 HTP/DSP 运行时：
   - 对齐 SDK `qnn_libs` 与板端 `/vendor/lib64`、`/vendor/lib/rfsa/adsp` 的版本。
   - 检查是否需要 signed skel、`testsig`、或板端 debug image 权限。
   - 重新跑 `qnn-platform-validator --backend dsp --testBackend`。

2. HTP 跑通后重测：
   - YOLOv5s INT8 `--use_dsp`
   - InceptionV3 INT8 `--use_dsp`
   - 记录吞吐、端到端延迟、初始化时间和稳定性。

3. 再做 CPU vs HTP 对比表：
   - FP32 CPU
   - INT8 CPU
   - INT8 HTP
