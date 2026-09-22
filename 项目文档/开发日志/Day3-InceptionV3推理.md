# Day 3：跑通 InceptionV3 图片分类

> 端侧 AI 测试（端测）学习 · 第三天
> 目标：TF 模型 → 转 DLC → CPU 推理 → 查看分类结果
> 日期：2026-06-30 凌晨
> **结果：全部完成 ✅**

---

## 一、环境快照

```
操作系统：Ubuntu 22.04.5 LTS
SNPE：    v2.22.6.240515184619_92920
Python：  3.10.12 (venv: ~/tools/venv/snpe/)
工作目录：${SNPE_ROOT}/examples/Models/InceptionV3/tensorflow
```

环境恢复：
```bash
source ~/snpe_env.sh
```

---

## 二、执行步骤

### 3.1 环境确认

```bash
source ~/snpe_env.sh
snpe-net-run --version
# → SNPE v2.22.6.240515184619_92920
```

### 3.2 下载模型和数据

```bash
export TENSORFLOW_HOME=$(python3 -c "import tensorflow as tf; print(tf.__path__[0])")
mkdir -p ~/tmpdir
python3 ${SNPE_ROOT}/examples/Models/InceptionV3/scripts/setup_inceptionv3.py -a ~/tmpdir -d
```

> ⚠️ **路径踩坑**：QNN SDK v2.22.6 目录名是 `InceptionV3`（大写 V），不是旧文档的 `inception_v3`。

脚本自动完成：
- 从 Google 下载 InceptionV3 frozen model（约 90MB tar.gz）
- 解压到 `tensorflow/` 目录
- 对 4 张示例图片做 crop + 转 raw + 生成 raw_list.txt

### 3.3 模型转换（TF → DLC）

```bash
cd ${SNPE_ROOT}/examples/Models/InceptionV3/tensorflow

snpe-tensorflow-to-dlc \
  -i inception_v3_2016_08_28_frozen.pb \
  -d input 1,299,299,3 \
  --out_node InceptionV3/Predictions/Reshape_1
# → INFO_CONVERSION_SUCCESS

mv inception_v3_2016_08_28_frozen.dlc inception_v3.dlc
# → 92MB DLC 文件
```

> 转换过程中大量 "Can only merge 1 encoding" WARNING 是正常的。

### 3.4 CPU 推理

```bash
snpe-net-run \
  --container inception_v3.dlc \
  --input_list ../data/cropped/raw_list.txt
```

输出到 `output/` 目录：
```
output/
├── SNPEDiag_0.log / SNPEDiag.log
├── Result_0/InceptionV3/Predictions/Reshape_1:0.raw
├── Result_1/InceptionV3/Predictions/Reshape_1:0.raw
├── Result_2/InceptionV3/Predictions/Reshape_1:0.raw
└── Result_3/InceptionV3/Predictions/Reshape_1:0.raw
```

4 张图片全部推理成功 ✅。

### 3.5 查看分类结果

```bash
python3 ../scripts/show_inceptionv3_classifications_snpe.py \
  -i ../data/cropped/raw_list.txt \
  -o output/ \
  -l ../data/imagenet_slim_labels.txt
```

> ⚠️ **脚本踩坑**：老版 `show_inceptionv3_classifications.py` 硬编码了 `Result_X/InceptionV3_Predictions_Reshape_1_0.raw`，新版 SDK 输出路径是 `Result_X/InceptionV3/Predictions/Reshape_1:0.raw`。用 `_snpe` 版脚本解决。

---

## 三、推理结果

### Top-1 结果

| # | 图片 | 预测类别 (ID/英文) | 置信度 | 评价 |
|---|------|-------------------|--------|------|
| 0 | notice_sign.jpg | 459 brass (铜器) | 13.0% | ❌ 误判，置信度低 |
| 1 | plastic_cup.jpg | 648 measuring cup (量杯) | **99.0%** | ✅ 非常精准 |
| 2 | trash_bin.jpg | 413 ashcan (垃圾桶) | **72.0%** | ✅ 正确 |
| 3 | chairs.jpg | 832 studio couch (沙发椅) | 38.1% | ⚠️ 接近但不精确 |

### Top-5 详细分析

#### notice_sign.jpg → brass 13.0% ❌

| 排名 | 类别 | 置信度 |
|------|------|--------|
| 1 | brass | 13.0% |
| 2 | street sign | 2.5% |
| 3 | rain barrel | 1.7% |
| 4 | rifle | 1.5% |
| 5 | park bench | 1.3% |

**分析：** 完全失败的预测。Top-1 仅 13%，Top-5 总和仅 20%，说明模型自己都不知道这是什么。1000 个类里随便选，强行选 brass 只是纹理相似。**这是最差的一类错误：低置信度 + 错误预测。** 实际应该判"street sign"（第二名），但模型不确信。

#### plastic_cup.jpg → measuring cup 99.0% ✅

| 排名 | 类别 | 置信度 |
|------|------|--------|
| 1 | measuring cup | **99.0%** |
| 2 | beaker | 0.2% |
| 3 | cocktail shaker | 0.1% |
| 4 | lotion | 0.03% |
| 5 | spatula | 0.02% |

**分析：** 教科书级好预测。Top-1 和 Top-2 差距达 **98.8 个百分点**，这是"一家独大"的分布。模型极度确信这是量杯，Top-5 全部是容器类，方向也对。**这是最好的预测：高置信度 + 正确 + 竞争类别合理。**

#### trash_bin.jpg → ashcan 72.0% ✅

| 排名 | 类别 | 置信度 |
|------|------|--------|
| 1 | ashcan | **72.0%** |
| 2 | milk can | 3.9% |
| 3 | saltshaker | 2.3% |
| 4 | oil filter | 1.9% |
| 5 | hamper | 1.6% |

**分析：** 可以接受的预测。72% 不算非常高，但第一名明显甩开第二名（68 个百分点差距）。Top-5 全是容器/桶类，语义方向正确。**这是中等置信度 + 正确预测，可接受。**

#### chairs.jpg → studio couch 38.1% ⚠️

| 排名 | 类别 | 置信度 |
|------|------|--------|
| 1 | studio couch | **38.1%** |
| 2 | seat belt | 3.5% |
| 3 | passenger car | 3.0% |
| 4 | home theater | 3.0% |
| 5 | velvet | 2.7% |

**分析：** 有趣的结果。沙发椅不算完全错（和椅子视觉相似），但置信度只有 38%，Top-5 出现了 seat belt/passenger car 等奇怪类别。模型有"坐着的东西"这个概念但表达不够精确。**这是低置信度 + 方向对但不精确，需要更多数据判断。**

### 质量评估总结

```
Top-1 准确率（4张）：2/4 = 50%
  - 注意：样本太小，这不代表模型真实水平
  - InceptionV3 在 ImageNet 50000 张验证集上的真实 top-1 约 78%

置信度分布：
  高置信 (>90%)： 1 张 ✅ (plastic_cup)
  中置信 (50-90%)：1 张 ✅ (trash_bin)
  低置信 (<50%)： 2 张   (notice_sign ❌ + chairs ⚠️)
```

> **关键认知：** 端测中的"质量"不只是看答对答错。置信度分布告诉你模型是"确定但错了"还是"不确定所以错了"——这两种错误的处理方式完全不同。确定但错 = 模型有系统性偏见；不确定所以错 = 模型特征不足，需要更多数据/更好训练。

---

## 四、Day 3 完成清单

| # | 任务 | 状态 |
|---|------|------|
| 1 | InceptionV3 模型下载 | ✅ ~90MB tar.gz |
| 2 | TF → DLC 转换 | ✅ 92MB DLC |
| 3 | snpe-net-run 推理 | ✅ 4/4 成功 |
| 4 | 查看分类结果 | ✅ 3/4 合理 |

---

## 五、踩坑记录

| # | 坑 | 现象 | 解决 |
|---|-----|------|------|
| 1 | **目录名不同** | 旧文档写 `inception_v3`，实际是 `InceptionV3`（大写 V） | 先 find 再 cd |
| 2 | **TENSORFLOW_HOME 路径不同** | 旧文档假设路径含 `/tensorflow_core`，实际是 keras 路径 | 用 `python3 -c "import tensorflow"` 自动获取 |
| 3 | **分类脚本版本不匹配** | 老 `show_inceptionv3_classifications.py` 硬编码路径找不到输出文件 | 改用 `_snpe.py` 版本 |
| 4 | **输出路径更深** | 新版输出在 `Result_X/InceptionV3/Predictions/Reshape_1:0.raw` | `_snpe.py` 脚本自动适配 |
| 5 | **`_snpe.py` 是 Python 2 代码** | `xrange` 在 Python 3 中不存在，NameError | `sed -i 's/xrange/range/g'` 修复 |
| 6 | **softmax 二次应用** | 写 Top-5 脚本时对 output raw 又做了一次 softmax，导致概率被"摊平"到 0.1% | 直接读 raw 值即为概率，不需要 softmax（模型输出层已包含 softmax） |

---

## 六、核心理解

```
原始 TF 模型 (.pb)
      │
      ▼ snpe-tensorflow-to-dlc
      │   ① 冻结图解析
      │   ② 算子映射（TF Op → SNPE Op）
      │   ③ 图优化
      │
SNPE DLC 模型 (.dlc) ──── 高通私有格式
      │
      ▼ snpe-net-run
      │   ① 加载 DLC 到内存
      │   ② 读取 .raw 输入 → 转 tensor
      │   ③ CPU 上逐层推理
      │   ④ 输出概率向量 → 写 .raw
      │
输出概率向量 (.raw)
      │
      ▼ Python 脚本
      │   ① 读取 float32 数组（已经是概率值，无需 softmax）
      │   ② 查标签文件 → 输出人类可读类别
      │
分类结果
```

**这就是端侧 AI 的"模型部署→推理"核心流水线。Day 4 的 YOLOv5 逻辑一样，只是模型结构和后处理更复杂。**
