# Day 4：YOLOv5 目标检测 + INT8 量化

> 端侧 AI 测试（端测）学习 · 第四天
> 目标：PyTorch → ONNX → DLC → INT8量化 → 推理 → 后处理画框
> 日期：2026-06-30
> **结果：全部完成 ✅**

---

## 一、环境快照

```
SNPE：    v2.22.6.240515184619_92920
Python：  3.10.12 (venv: ~/tools/venv/snpe/)
工作目录：~/day4-yolo/
测试图：  bus.jpg (810x1080, COCO 公交场景)
```

---

## 二、执行步骤

### 4.1 YOLOv5 → ONNX

```bash
cd ~/day4-yolo
git clone https://github.com/ultralytics/yolov5.git
cd yolov5 && pip install -r requirements.txt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python3 export.py --weights yolov5s.pt --batch 1 --include onnx
# → yolov5s.onnx (28.3MB, opset 18)
```

> ⚠️ **坑1：numpy 不兼容** — pip 升级到 numpy 2.2，pandas 报错 `numpy.dtype size changed`。`pip install numpy==1.26.4` 修复。
> ⚠️ **坑2：opencv 缺图形库** — opencv-python 4.13 报 `libGL.so.1: cannot open`，`sudo apt install -y libgl1-mesa-glx` 修复。

### 4.2 前处理

按实践文档 6.3 节逻辑改写，加 resize 640x640：

```bash
python3 02_scripts/preprocess.py 03_input/bus.jpg
# → 03_input/bus.raw (640x640x3, float32)
```

核心逻辑：`cv2.read → resize(640,640) → (img - 128) / 128 → float32 → .raw`

### 4.3 ONNX → DLC 转换

```bash
snpe-onnx-to-dlc --input_network yolov5/yolov5s.onnx --output_path 01_models/yolov5s.dlc
# → 01_models/yolov5s.dlc (29MB)
```

> ⚠️ **坑3：ONNX opset 不兼容** — torch 导出的是 opset 18，SNPE v2.22.6 最高支持 13。
> **修复**：Python 脚本手动修改 ONNX protobuf：降 opset → 13，修复 Reshape allowzero → 0，删除 Resize 的 antialias/keep_aspect_ratio_policy 属性。

### 4.4 INT8 量化

校准集：200 张 bus.jpg 随机变体（缩放/裁剪/亮度）

```bash
snpe-dlc-quantize \
  --input_dlc 01_models/yolov5s.dlc \
  --input_list 03_input/calib_raws/raw_list.txt \
  --output_dlc 01_models/yolov5s_quantized.dlc
```

| | FP32 | INT8 | 缩减 |
|------|------|------|------|
| 模型大小 | 29MB | 7.2MB | **75%** |
| 推理耗时 | ~7s/张 | ~7s/张 | CPU 上无差异 |
| 检测精度 | baseline | **几乎无损失** | <0.02 置信度 |

> CPU 上速度无差异是正常的——Intel/AMD CPU 的 INT8 指令加速需要特定编译优化，SDK 默认 FP32 路径。在 DSP/HTP 上 INT8 会快 2-4x。

### 4.5 SNPE 推理

新版 YOLOv5 导出后检测结果在单个 output0 (25200×85)，不需要 `--set_output_tensors`：

```bash
snpe-net-run --container 01_models/yolov5s_quantized.dlc --input_list 03_input/target_raw_list.txt --output_dir /tmp/output_quantized
```

### 4.6 后处理

纯 Python：numpy 解码 + NMS + cv2 画框。

```bash
python3 02_scripts/postprocess.py output0.raw bus.jpg result.jpg
```

---

## 三、检测结果

### FP32 模型

| 物体 | 置信度 | 边界框 |
|------|--------|--------|
| person | 1.00 | [44,232,158,553] |
| person | 0.99 | [1,325,68,528] |
| person | 0.97 | [180,252,268,510] |
| person | 0.97 | [548,298,641,515] |
| skateboard | 0.99 | [526,489,638,524] |

### INT8 量化模型

| 物体 | 置信度 | 边界框 |
|------|--------|--------|
| person | 1.00 | [40,231,161,552] |
| person | 0.99 | [529,329,641,522] |
| person | 0.98 | [1,323,70,530] |
| person | 0.95 | [178,234,271,505] |
| skateboard | 0.99 | [524,488,638,523] |

**量化精度损失：几乎为零。**

---

## 四、工作目录结构

```
~/day4-yolo/
├── 01_models/                   ← 模型
│   ├── yolov5s.onnx             28.3MB
│   ├── yolov5s.dlc              29MB
│   └── yolov5s_quantized.dlc    7.2MB
│
├── 02_scripts/                  ← 可复用脚本
│   ├── preprocess.py            前处理
│   └── postprocess.py           后处理（NMS+画框）
│
├── 03_input/                    ← 输入数据
│   ├── bus.jpg                  测试原图
│   ├── bus.raw                  前处理后 raw
│   ├── target_raw_list.txt     raw 路径列表
│   ├── calib_images/            200 张校准图
│   └── calib_raws/              200 张校准 raw + raw_list.txt
│
├── 04_output/                   ← 推理产物
│   ├── result_by_snpe.jpg       FP32 检测结果
│   └── quantized/
│       └── result_quantized.jpg INT8 检测结果
│
├── 05_temp/                     ← 中间文件
│   └── yolov5s_fixed.onnx       opset 降级后的 ONNX
│
└── yolov5/                      ← YOLOv5 源码
    ├── yolov5s.pt               15MB 原始权重
    └── export.py
```

---

## 五、Day 4 完成清单

| # | 任务 | 状态 |
|---|------|------|
| 1 | YOLOv5s 导出 ONNX | ✅ 28.3MB |
| 2 | 前处理生成 .raw | ✅ resize + 归一化 |
| 3 | ONNX → DLC | ✅ 29MB |
| 4 | INT8 量化 | ✅ 29MB → 7.2MB (-75%) |
| 5 | SNPE 推理 | ✅ FP32 + INT8 均成功 |
| 6 | 后处理画框 | ✅ 5 物体检出，量化无损失 |
| 7 | 拍 Day 4 备份 | ✅ ubuntu-snpe-day4-clean.utm |

---

## 六、备份快照

```bash
# Mac 终端（虚拟机关机后）
cp -r ~/Library/Containers/com.utmapp.UTM/Data/Documents/ubuntu-snpe.utm \
      ~/Documents/端测/ubuntu-snpe-day4-clean.utm
```

| 备份 | 文件 | 状态 |
|------|------|------|
| Day 1 | ubuntu-snap-backup-day1.utm | ✅ |
| Day 2 | ubuntu-snpe-day2-clean.utm | ✅ |
| Day 4 | ubuntu-snpe-day4-clean.utm | ✅ |

---

## 六、踩坑记录

| # | 坑 | 现象 | 解决 |
|---|-----|------|------|
| 1 | **numpy 2.x 不兼容** | pip 装 ultralytics 升级 numpy→2.2，pandas `numpy.dtype size changed` | `pip install numpy==1.26.4` |
| 2 | **opencv 缺 libGL** | `libGL.so.1: cannot open shared object file` | `sudo apt install -y libgl1-mesa-glx` |
| 3 | **ONNX opset 18 不兼容** | SNPE 只支持 opset ≤13，Resize/Reshape 属性报错 | Python 脚本降 opset+修复属性 |
| 4 | **onnx 版本反复切换** | ultralytics 拉 onnx→1.22 导致 SNPE 报 `AttributeProto` 错误 | `pip install onnx==1.13.0` 才能转 DLC |
| 5 | **protobuf 版本冲突** | tensorflow 要 <3.20，ultralytics 要 ≥4.25 | 转 DLC 时用 onnx 1.13+protobuf 3.20，导出 ONNX 时升回去 |
| 6 | **新版 YOLOv5 输出结构变化** | 旧文档的 3 个检测头输出名变成了单个 output0 | 直接完整推理，不需 `--set_output_tensors` |
| 7 | **图像尺寸不匹配** | bus.jpg 810x1080，YOLOv5 要 640x640 | preprocess.py 加 `cv2.resize(img, (640,640))` |

---

## 七、核心理解

### 量化到底是什么

```
FP32 模型
  每层权重: -0.523 → 用 32 位浮点数存
  每层激活: 3.14159 → 用 32 位浮点数传递

量化过程:
  1. 用校准集跑一遍 FP32 模型
  2. 统计每层 tensor 的 min/max 范围
  3. 算出 scale 和 offset，把 float 映射到 uint8 [0,255]
     float = scale × (uint8 - offset)
  4. 把模型权重替换成 uint8 + scale/offset

INT8 模型
  每层权重: -0.523 → 用 8 位整数存（4x 压缩）
  每层激活: 3.14159 → 用 8 位整数传递
  推理时: 整数运算 → 乘 scale 还原 → 继续传
```

### 量化为什么可能损失精度
- 浮点数有 2^32 个取值，整数只有 2^8=256 个取值
- 有些层的取值范围很大（如 softmax 后的概率分布很"尖"）
- 映射到 256 个格子里会丢掉细微差异
- 但现代网络有很多冗余，INT8 通常只丢 <1% 精度

### 端测部署的完整流程

```
训练模型 (.pt/.pb)
      │
      ▼ 导出
ONNX 模型 (.onnx)       ← 通用中间格式
      │
      ▼ snpe-onnx-to-dlc
DLC 模型 (.dlc)         ← 高通私有格式
      │
      ▼ snpe-dlc-quantize
量化 DLC (.dlc)         ← 端侧优化的最终产物
      │
      ▼ snpe-net-run
推理结果 (.raw)         ← 原始输出
      │
      ▼ 后处理
最终结果 (画框/分类标签/...)
```

---

## 八、Day 4 vs Day 3 对比

| | Day 3 (InceptionV3) | Day 4 (YOLOv5) |
|------|---------------------|------------------|
| 任务 | 图片分类 | 目标检测 |
| 模型 | InceptionV3 (92MB) | YOLOv5s (29MB) |
| 输出 | 1 个 1000 维向量 | 1 个 25200×85 张量 |
| 量化 | 无 | **INT8 (29MB→7.2MB)** |
| 后处理 | argmax 查标签 | 解码+NMS+画框 |
| 踩坑数 | 6 | 7 |
| 难度 | ★★☆ | ★★★★ |
