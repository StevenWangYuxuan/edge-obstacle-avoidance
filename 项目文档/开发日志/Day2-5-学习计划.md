# Day 2 ~ Day 5：安装 SNPE 到完成端测入门

> 端侧 AI 测试（端测）学习 · 后续计划
> 前置：Day 1 已搭好 Ubuntu 20.04 虚拟机（`ubuntu-snpe`，IP 192.168.64.3，用户 steven）
> 参考文档：《基于高通SNPE的深度学习推理与应用实践 V1.1》（以下简称"实践文档"）

---

## 总览

| 天数 | 主题 | 目标 | 完成标志 |
|------|------|------|----------|
| Day 2 | 安装 SNPE | 装好 SNPE SDK + Python 环境 | `snpe-net-run --version` 出版本号 |
| Day 3 | 跑通 InceptionV3 | 模型转换 + 主机 CPU 推理 | 输出图片分类结果 |
| Day 4 | 跑通 YOLOv5 + 量化 | 转换 + 量化 + 推理 + 后处理 | 输出带检测框的图片 |
| Day 5 | 学会看性能数据 | 诊断 + 基准测试 + 算指标 | 能说出 inf/s、内存占用 |

> 每天 1 个明确目标，做完再做下一个。预计每天 2~4 小时（含踩坑时间）。

---

# Day 2：安装 SNPE

## 目标
在 Ubuntu 虚拟机里装好高通 SNPE SDK 和 Python 环境，跑出版本号。

## 详细步骤

### 2.1 启动虚拟机并连接
```bash
# Mac 上
# 1. UTM 启动 ubuntu-snpe
# 2. Mac 终端
ssh steven@192.168.64.3
# 连不上就 UTM 窗口登录，ip addr show 查新 IP
```

### 2.2 安装 QPM3（高通包管理器）
QPM3 是下载 SNPE 的入口，必须登录高通账号。

```bash
# 下载 QPM3 的 deb 包（链接从高通官网获取）
# 官网：https://qpm.qualcomm.com/#/main/tools/details/QPM3
# 选 Linux 版本下载，得到类似：
#   QualcommPackageManager3.x.x.x.Linux-x86.deb

# 安装
sudo dpkg -i QualcommPackageManager3.x.x.x.Linux-x86.deb
```

> ⚠️ 坑点：deb 包要在虚拟机里下载或从 Mac 传进去。若虚拟机里下载慢，可在 Mac 下载后用 `scp` 传进虚拟机：
> ```bash
> # Mac 终端
> scp ~/Downloads/QualcommPackageManager3.x.x.x.Linux-x86.deb steven@192.168.64.3:~/
> ```

### 2.3 登录 QPM3 并下载 SNPE
```bash
# 启动 QPM3（图形界面，但虚拟机是无界面的 Server 版，可能要用命令行方式）
qpm3
# 登录高通账号
# 选择 "Qualcomm Neural Processing SDK" → Linux 版本 → Extract
```

> ⚠️ Server 版无图形界面，QPM3 可能有命令行模式（`qpm3 --help` 查看）。如果图形界面起不来，需要查 QPM3 CLI 用法，或临时装个轻量桌面。这步可能踩坑，到时具体处理。

### 2.4 确认 SNPE 安装路径
```bash
# 预期路径类似：
ls /opt/qcom/aistack/snpe/
# 应该看到版本号目录，如 2.15.0.230926
```

### 2.5 配置环境变量
```bash
# 设置 SNPE 根目录（路径根据实际版本号替换）
export SNPE_ROOT=/opt/qcom/aistack/snpe/2.15.0.230926

# 配置 SNPE 运行环境
source ${SNPE_ROOT}/bin/envsetup.sh

# 验证
snpe-net-run --version
```

✅ **完成标志**：`snpe-net-run --version` 输出版本号

### 2.6 安装 Python3.8 虚拟环境
```bash
# 装 python3.8 虚拟环境工具
sudo apt install -y python3.8-venv

# 创建虚拟环境
python3.8 -m venv ~/tools/venv/python3.8/

# 激活
source ~/tools/venv/python3.8/bin/activate
# 提示符前出现 (python3.8) 表示成功

python --version  # 应显示 3.8.x
```

> ⚠️ Ubuntu 重启后需重新 `source` 激活虚拟环境。

### 2.7 安装 AI 框架依赖
```bash
# 升级 pip
python3 -m pip install --upgrade pip

# 检查 SNPE 的 python 依赖
${SNPE_ROOT}/bin/check-python-dependency

# 检查 linux 依赖
sudo bash ${SNPE_ROOT}/bin/check-linux-dependency.sh

# 安装 TensorFlow（转换 TF 模型用）
pip3 install tensorflow==2.10.1

# 安装 ONNX（转换 ONNX 模型用）
pip3 install onnx==1.11.0

# 安装 PyTorch（YOLOv5 导出用，CPU 版即可）
pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ pip 下载慢可换国内源：
> ```bash
> pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.10.1
> ```

### 2.8 重启后快速恢复环境（保存为脚本）
```bash
# 每次重启后执行这 4 条即可恢复环境
source ~/tools/venv/python3.8/bin/activate
export SNPE_ROOT=/opt/qcom/aistack/snpe/2.15.0.230926
source ${SNPE_ROOT}/bin/envsetup.sh
export TENSORFLOW_HOME=<tensorflow 路径，用 pip show tensorflow 查>
```

> 建议把这 4 条写进 `~/snpe_env.sh`，以后 `source ~/snpe_env.sh` 一键恢复。

## Day 2 完成清单
- [ ] QPM3 装好，能登录高通账号
- [ ] SNPE SDK 下载安装到 `/opt/qcom/aistack/snpe/`
- [ ] `snpe-net-run --version` 出版本号
- [ ] Python3.8 虚拟环境建好
- [ ] TensorFlow / ONNX / PyTorch 装好
- [ ] 环境恢复脚本写好

---

# Day 3：跑通 InceptionV3 图片分类 ✅

> 完成日期：2026-06-30 凌晨

## 目标
把一个 TensorFlow 训练好的图片分类模型，转成 SNPE 的 DLC 格式，在主机 CPU 上推理出分类结果。

## 详细步骤（已执行）

### 3.1 确认环境 ✅
```bash
source ~/snpe_env.sh
snpe-net-run --version   # SNPE v2.22.6.240515184619_92920
```

### 3.2 下载 InceptionV3 模型和数据集 ✅
```bash
export TENSORFLOW_HOME=$(python3 -c "import tensorflow as tf; print(tf.__path__[0])")
mkdir -p ~/tmpdir
python3 ${SNPE_ROOT}/examples/Models/InceptionV3/scripts/setup_inceptionv3.py -a ~/tmpdir -d
```

> ⚠️ **路径踩坑**：QNN SDK v2.22.6 目录名是 `InceptionV3`（大写 V），不是旧 SNPE 文档的 `inception_v3`。

产物：
- 模型：`${SNPE_ROOT}/examples/Models/InceptionV3/tensorflow/inception_v3_2016_08_28_frozen.pb`
- 数据：`${SNPE_ROOT}/examples/Models/InceptionV3/data/cropped/`（4 张 raw + raw_list.txt）
- 标签：`${SNPE_ROOT}/examples/Models/InceptionV3/data/imagenet_slim_labels.txt`

### 3.3 模型转换（TF → DLC）✅
```bash
cd ${SNPE_ROOT}/examples/Models/InceptionV3/tensorflow

snpe-tensorflow-to-dlc \
  -i inception_v3_2016_08_28_frozen.pb \
  -d input 1,299,299,3 \
  --out_node InceptionV3/Predictions/Reshape_1

mv inception_v3_2016_08_28_frozen.dlc inception_v3.dlc
# → 92MB DLC
```

> ⚠️ 转换过程中大量的 "Can only merge 1 encoding" WARNING 是正常的，不影响。最后 `INFO_CONVERSION_SUCCESS` 即成功。

### 3.4 用 SNPE 推理 ✅
```bash
snpe-net-run \
  --container inception_v3.dlc \
  --input_list ../data/cropped/raw_list.txt
```
输出在 `output/`，4 张图片全部成功推理。

### 3.5 查看分类结果 ✅
```bash
# ⚠️ 用 _snpe 版脚本，老版路径不匹配
python3 ../scripts/show_inceptionv3_classifications_snpe.py \
  -i ../data/cropped/raw_list.txt \
  -o output/ \
  -l ../data/imagenet_slim_labels.txt
```

> ⚠️ **脚本踩坑**：老版 `show_inceptionv3_classifications.py` 硬编码了旧路径格式，新版 SDK 输出路径是 `output/Result_X/InceptionV3/Predictions/Reshape_1:0.raw`。用 `_snpe.py` 版本解决。

实际输出文件路径：`output/Result_X/InceptionV3/Predictions/Reshape_1:0.raw`

## 推理结果

| 图片 | 预测类别 | 置信度 | 评价 |
|------|---------|--------|------|
| notice_sign.jpg | 459 brass (铜器) | 13.0% | ❌ |
| plastic_cup.jpg | 648 measuring cup (量杯) | **99.0%** | ✅ |
| trash_bin.jpg | 413 ashcan (垃圾桶) | **72.0%** | ✅ |
| chairs.jpg | 832 studio couch (沙发椅) | 38.1% | ⚠️ |

✅ **完成标志**：模型转换→推理→分类结果全链路跑通。

## Day 3 完成清单
- [x] InceptionV3 模型下载成功
- [x] 成功转换为 DLC 格式（92MB）
- [x] `snpe-net-run` 推理无报错（4/4）
- [x] 看到正确的分类结果（3/4 合理）

---

# Day 4：跑通 YOLOv5 目标检测 + 量化 ✅

> 完成日期：2026-06-30

## 目标
端测核心环节：把 PyTorch 模型导出 ONNX → 转 DLC → **量化（INT8）** → 推理 → 后处理画检测框。

## 详细步骤（已执行）

### 4.1 准备 YOLOv5 模型 ✅
```bash
cd ~/day4-yolo && git clone https://github.com/ultralytics/yolov5.git
cd yolov5 && pip install -r requirements.txt
# 升级导致的 numpy 2.x 不兼容 → pip install numpy==1.26.4 修复
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python3 export.py --weights yolov5s.pt --batch 1 --include onnx
# → yolov5s.onnx (28.3MB, opset 18)
```
> ⚠️ `numpy 2.x` 导致 pandas 报错 → 降回 1.26.4

### 4.2 前处理 ✅
按文档 6.3 节 `pre()` 函数改写，加 resize 640x640：
- cv2 读图 → resize(640,640) → (img - 128) / 128 → float32 → .raw
- 脚本：`~/day4-yolo/02_scripts/preprocess.py`

### 4.3 ONNX → DLC ✅
```bash
snpe-onnx-to-dlc --input_network yolov5/yolov5s.onnx --output_path yolov5s.dlc
```
> ⚠️ 新 torch 导出 opset 18 不兼容：手动修复 Reshape allowzero + Resize 属性 + 降 opset → 13

### 4.4 INT8 量化 ✅
```bash
snpe-dlc-quantize --input_dlc yolov5s.dlc --input_list calib_raws/raw_list.txt --output_dlc yolov5s_quantized.dlc
```
| | FP32 | INT8 | 缩减 |
|------|------|------|------|
| 模型大小 | 29MB | 7.2MB | **75%** |

### 4.5 SNPE 推理 ✅
```bash
snpe-net-run --container yolov5s_quantized.dlc --input_list target_raw_list.txt --output_dir output_quantized
```
> 新版 YOLOv5 导出后 25200 detections 在单个 output0 输出，不需要 --set_output_tensors

### 4.6 后处理 ✅
纯 Python：numpy 解码 × NMS × cv2 画框，脚本 `~/day4-yolo/02_scripts/postprocess.py`

推理结果（bus.jpg, INT8）：
| 物体 | 置信度 |
|------|--------|
| person | 1.00 |
| person | 0.99 |
| person | 0.98 |
| person | 0.95 |
| skateboard | 0.99 |

## 量化前后对比

| 指标 | FP32 | INT8 | 差异 |
|------|------|------|------|
| 模型大小 | 29MB | 7.2MB | -75% |
| person 检测数 | 4 | 4 | 相同 |
| skateboard | 1 | 1 | 相同 |
| 置信度差异 | - | - | <0.02 |

**量化精度损失几乎为零。**

## 工作目录

```
~/day4-yolo/
├── 01_models/          yolov5s.onnx / yolov5s.dlc / yolov5s_quantized.dlc
├── 02_scripts/         preprocess.py / postprocess.py
├── 03_input/           bus.jpg / bus.raw / calib_images/ / calib_raws/
├── 04_output/          result_by_snpe.jpg / quantized/
├── 05_temp/            yolov5s_fixed.onnx
└── yolov5/             YOLOv5 源码仓库
```

## Day 4 完成清单
- [x] YOLOv5s 导出 ONNX（28.3MB）
- [x] 前处理代码跑通，生成 .raw
- [x] ONNX → DLC 转换（29MB）
- [x] INT8 量化（29MB → 7.2MB, -75%）
- [x] SNPE 推理出检测结果（5 物体）
- [x] 后处理画出检测框（量化后精度无损失）

---

# Day 5：学会看性能数据（端测入门）

## 目标
掌握 SNPE 的性能测量工具，能算出推理速度和资源占用，对照端测指标体系。这一天才是真正意义上的"端测"。

## 详细步骤

### 5.1 用 snpe-diagview 看推理耗时
```bash
# 解析 SNPEDiag.log，输出推理时间
snpe-diagview --input_log output/SNPEDiag.log
# 会显示 Total Inference Time: xxx us
```

### 5.2 用 snpe_bench.py 跑基准测试

#### 配置 json 文件
位置：`${SNPE_ROOT}/benchmarks/SNPE/`，参考 `config_help.json` 写一个 `sample.json`：
```json
{
    "Name": "inceptionv3",
    "HostRootPath": "inceptionv3",
    "HostResultsDir": "inceptionv3/results",
    "DevicePath": "/data/local/tmp/inception_v3/bm",
    "Devices": ["<adb 设备号，主机推理可填 localhost>"],
    "HostName": "localhost",
    "Runs": 2,
    "Model": {
        "Name": "Inception_v3",
        "Dlc": "../../examples/Models/inception_v3/tensorflow/inception_v3.dlc",
        "InputList": "../../examples/Models/inception_v3/data/cropped/raw_list.txt",
        "Data": ["../../examples/Models/inception_v3/data/cropped/"]
    },
    "Runtimes": ["CPU"],
    "Measurements": ["timing", "mem"]
}
```

#### 运行
```bash
python3 snpe_bench.py -t android-aarch64 -c sample.json
# 主机 x86 推理用对应的 target 类型
```

#### 查看结果
结果在 `${SNPE_ROOT}/benchmarks/SNPE/inceptionv3/results/` 下的 `benchmark_stats_inceptionv3.csv`。

#### 算 inf/s
```
inf/s = 1000000 / (Total Inference Time avg(us) / Runs 次数)
例：avg 1574us, 2 runs → 1000000/(1574/2) = 1270.65 inf/s
```

### 5.3 用 snpe-throughput-net-run 测吞吐量
```bash
snpe-throughput-net-run \
  --container inception_v3.dlc \
  --duration 5 \
  --use_cpu \
  --perf_profile high_performance
```
> `--use_cpu` 可换 `--use_gpu` / `--use_dsp`（DSP/HTP 需真机）
> `--perf_profile` 可选：balanced / default / high_performance / sustained_high_performance / burst / power_saver

### 5.4 对照端测指标体系整理结果

把 Day 3、Day 4 的模型结果填入下表（参考《端侧AI模型全景与测试方案》四维度）：

| 维度 | 指标 | InceptionV3 | YOLOv5s |
|------|------|-------------|---------|
| 性能 | 推理耗时 (ms) | | |
| 性能 | 吞吐 (inf/s) | | |
| 资源 | 模型大小 (MB) | | |
| 资源 | 内存占用 (MB) | | |
| 质量 | 分类/检测准确率 | | |
| 稳定性 | 连续推理多次耗时波动 | | |

✅ **完成标志**：你能说出"YOLOv5s 在主机 CPU 上每秒能推理多少次、模型多大、占多少内存"

## Day 5 完成清单
- [ ] snpe-diagview 能看推理耗时
- [ ] snpe_bench.py 跑出 CSV 结果
- [ ] 会算 inf/s
- [ ] snpe-throughput-net-run 测出吞吐
- [ ] 整理出一张性能对比表

---

## 阶段总结：5 天后的成果

走完这 5 天，你将完整掌握：

1. **环境搭建**：Ubuntu + SNPE + Python 全套环境（Day 1-2）
2. **模型部署流程**：训练模型 → 转换 → 量化 → 推理 → 后处理（Day 3-4）
3. **端测指标测量**：性能、资源、质量、稳定性四维度（Day 5）

这 70% 的流程在虚拟机上跑通后，剩下 30%（HTP/DSP 真机推理）等以后拿到美格模组（如 SNM970/972 QCS8550），只需把模型 push 上去换 `--use_dsp` 参数即可，流程你已经会了。

---

## 通用注意事项

1. **每一步都可能踩坑**：联网、依赖版本、路径、权限问题。遇到报错把完整输出贴给 Claude，别慌
2. **善用快照/备份**：每个 Day 开始前，可在 UTM 里关机做个备份点，搞坏了能还原
3. **环境变量易失效**：重启后 SNPE 环境会丢，用 Day 2.8 的脚本一键恢复
4. **换国内源**：apt 用阿里云、pip 用清华源，能省大量时间
5. **参考文档**：主要看《基于高通SNPE的深度学习推理与应用实践 V1.1》，命令基本照搬
