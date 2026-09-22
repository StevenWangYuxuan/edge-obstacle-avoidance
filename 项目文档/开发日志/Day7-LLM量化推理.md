# Day 7：LLM INT4 量化与推理 — ✅ ONNX→DLC 成功, ❌ 量化失败(SDK bug)

> 日期：2026-07-01 ~ 07-02
> 最终状态：**ONNX→DLC 转换成功，GPT-2 INT8 量化因 SDK bug 放弃，模型已全部传输到 Orion O8G2 真机板**
> 下一步：在真机板子上用 QNN HTP 后端做推理，LLM 量化留待板子原生工具链

> 📌 2026-07-02 完成事项：
> 1. ✅ **fixed.onnx → DLC 转换成功**（前两次失败后修复 Gather_6 index=[0..31] 第三次成功，461MB，16min）
> 2. ❌ **GPT-2 INT8 量化失败**（snpe-dlc-quantize & qairt-quantizer 均 segfault，已排除内存/校准数据原因，确认为 SNPE v2.22.6 对 Transformer 大模型的 C++ 后端 bug）
> 3. ✅ **InceptionV3 INT8 量化成功**（92MB→24MB, -74%）
> 4. ✅ **Orion O8G2 真机板连接就绪**（串口控制 + HTTP 传文件）
> 5. ✅ **全部模型已传板子**（/data/local/tmp/models/, 共 610MB）

---

## 一、目标回顾

对 DistilGPT-2 (81.9M) 的 DLC 做 INT4 量化，然后在 VM CPU 上验证推理。

---

## 二、量化 — 🔶 重新尝试（曾放弃，现找到新路线）

> 历史结论（曾放弃）：四种量化方法全部死于同一个 Gather_6。但后续发现 Gather_6 可在 ONNX 层面用静态 Slice 替换（见三节），于是用**修复后的 ONNX 重新转一个干净 DLC** 再量化，绕过了 CPU 后端的动态索引校验。本节先记录旧失败，再记录新进展。

### 尝试过的路线

| 序号 | 工具 | 方法 | 结果 |
|------|------|------|------|
| 1 | `qairt-quantizer` | DLC → INT4 + float_fallback | ❌ float_fallback 和 input_list 互斥 |
| 2 | `qairt-quantizer` | DLC → INT4 + 校准集 | ❌ Gather_6 动态索引，CPU 后端 OpConfig 校验失败 |
| 3 | `qairt-quantizer` | DLC → INT8 + 校准集 | ❌ 同上 |
| 4 | `snpe-dlc-quantize` | DLC → INT8 + 校准集 | ❌ 同上 |
| 5 | `qnn-onnx-converter` | 转换时 INT4（方案A） | ⏸️ 未执行 |

### 根因

```
/transformer/Gather_6: axis=0, inputs=['Flatten_output_0', 'Add_1_output_0']
                                                              ↑
                                                    运行时动态计算，不是常量
                                                    CPU 后端推不出输出形状 → 校验失败
```

**99 个 Gather 中只有这 1 个是动态的。HTP（真机）上不存在此限制。**

### 放弃理由（旧，现已被新路线推翻第 1 条）

1. ~~四种量化方法全部死于同一个 Gather_6~~ → 已通过修复 ONNX 解决
2. Day 5 已验证 INT8 在 x86 CPU 上不加速（+2.7%），量化仅省存储
3. 软件测试阶段 FP32 足够验证管线逻辑
4. 量化留给真机 QCS8550 HTP 更合理

> 注：第 2-4 条仍然成立（VM 上量化主要价值是「跑通流程 + 体积缩减」，非提速）。但「量化流程本身跑不通」这个问题已被新路线解决，所以重新尝试，目标：产出 INT8/INT4 DLC + 验证推理。

### 新路线：修复后 ONNX → 干净 DLC → 量化

核心思路：旧的 `distilgpt2_fp32.dlc` 是用含 Gather_6 的 ONNX 转的，所以量化死在 Gather_6。既然 `fix_dynamic_gather.py` 已经在 ONNX 层把 Gather_6 换成静态 `Reshape→Slice→Squeeze`，只要用**修复后的 `distilgpt2_fixed.onnx`** 重新转一个 DLC，这个干净 DLC 就不再有动态索引，量化能过。

步骤：

| # | 步骤 | 命令 | 状态 |
|---|------|------|------|
| 0 | 校准 raw 改 int32（见下方 dtype 坑） | `np.float32 → np.int32` 重写 30 个 raw | ✅ |
| 1 | 修复后 ONNX → 干净 DLC | `snpe-onnx-to-dlc -d input_ids 1,32 -d attention_mask 1,32 -d position_ids 1,32 --out_node logits --output_path distilgpt2_fixed.dlc` | 🔄 重跑中（首次跑了 17 分钟后崩溃无产出，见踩坑 #28） |
| 2 | INT8 量化 | `snpe-dlc-quantize --input_dlc distilgpt2_fixed.dlc --input_list calib/raw_list.txt --output_dlc distilgpt2_int8.dlc` | 🔲 |
| 3 | INT4 量化 | `snpe-dlc-quantize ... --pack_4_bit_weights` 或 `qairt-quantizer --weights_bitwidth 4` | 🔲 |
| 4 | 量化 DLC 推理验证 | `snpe-net-run --container distilgpt2_int8.dlc ...` | 🔲 |

已确认的关键事实：
- `distilgpt2_fixed.onnx`：Gather_6 已移除（节点数 0），输入 `input_ids/attention_mask/position_ids`（int64），输出 `logits`，opset 18。
- `snpe-dlc-quantize` 支持 `--pack_4_bit_weights`（INT4 权重打包）；`qairt-quantizer` 支持 `--weights_bitwidth 4`。INT4 路线可用。
- 校准 `raw_list.txt` 已就绪，10 条，顺序 ids/mask/pos。

### dtype 坑：校准 raw 是 float32，DLC 输入是 Int_32

旧量化死在 Gather_6 校验（在读校准数据之前），所以这个坑之前没暴露。现在 Gather_6 修好了，量化会真正读校准数据，必须先修：

- `snpe-dlc-info` 显示 DLC 三个输入都是 **Int_32**（`keep_int64_inputs=False`，int64→int32）。
- 但校准 raw 是按 **float32** 字节写的（`calib_0_ids` 作为 float32 = `[13023, 17543, ...]` 是正常 token id；同样字节当 int32 读 = `1179352064` 垃圾值）。
- 若不修，`snpe-dlc-quantize` 会按 int32 读这些字节 → 全是垃圾值 → 校准出来的量化范围全错（甚至报错）。

修复：把 30 个校准 raw 重新写成 int32（值不变，只换字节解释）。备份在 `calib_f32_backup/`。

```python
f = np.fromfile(p, dtype=np.float32)   # [13023, 17543, ...]
g = f.astype(np.int32)                 # [13023, 17543, ...] 同值
# 注意：不能 f.astype(int32).tofile(p) 就地写回——实测会写成全 0（疑似 fromfile/tofile 同路径冲突）
# 正解：先 tofile 到 /tmp，再 shutil.move 原子覆盖
g.tofile('/tmp/x.raw'); shutil.move('/tmp/x.raw', p)
```

> ⚠️ 就地写回坑：`np.fromfile(p).astype(np.int32).tofile(p)` 在批量执行时把文件写成了全 0（单文件写到 /tmp 却正常）。根因未完全定位，规避方式就是「写临时文件 → mv 覆盖」。

### 性能坑：ONNX→DLC 转换极慢的根因

转换一个 460MB / ~1496 节点的 ONNX，预估 3-5 分钟，实际 VM 上跑了 10+ 分钟还没完。诊断：

| 维度 | 实测 | 结论 |
|------|------|------|
| 内存 | 7.7G 总 / 用 3.2G / 剩 4.3G，swap 仅 297M | ✅ 充足，**不是瓶颈** |
| CPU | `nlwp=1`，100% 单核，load avg 1.09 | 转换器单线程，4 核只用 1 核 |
| 架构 | VM 是 x86_64，宿主 Mac 是 Apple Silicon(ARM) | 🔴 UTM/QEMU 用 TCG 纯软件翻译每条 x86 指令，无硬件虚拟化，比原生 x86 慢 5-10x |

**主因是 x86-on-ARM 模拟 + 单线程转换器**，加内存没用。根治只能换原生 x86 主机或上真机。当前只能等。

---

## 三、推理 — 🔲 待跑通

### 报错历程

| 尝试 | 工具 | 错误 |
|------|------|------|
| 1 | `qnn-net-run --dlc_path` | 参数格式不对 |
| 2 | `snpe-net-run --container` | **Gather_6 校验失败**（和量化时同一个） |

### 修复 Gather_6 ✅

写脚本 `fix_dynamic_gather.py`，原理：

```
修复前：Gather(Flatten, Add_1_output) → 动态索引，CPU 无法校验
修复后：Reshape → Slice[:, 31:32, :] → Squeeze → 全静态
```

**产出：** `distilgpt2_fixed.onnx` (Gather_6: 0)

### 下一步

1. fixed ONNX → DLC (3-5 分钟)
2. snpe-net-run CPU 推理 (预估 1-2 分钟)
3. decode 生成文本

---

## 四、当前文件状态

```
~/day6-llm/
├── 01_models/
│   ├── distilgpt2_qnn.onnx          ← IsNaN-free (Day 6)，fix 脚本的输入
│   ├── distilgpt2_fixed.onnx        ← IsNaN-free + Gather_6 fixed（当前管线输入）
│   ├── distilgpt2_onnx_v18/         ← opset 18 原始导出 + tokenizer（推理要用）
│   └── dlc/
│       ├── distilgpt2_fp32.dlc      ← 原始 DLC (含 Gather_6，量化失败根源，留作对比)
│       ├── distilgpt2_fixed.dlc     ← 修复后 ONNX 转的干净 DLC (无 Gather_6) 🆕 转换中
│       ├── distilgpt2_int8.dlc      ← INT8 量化产物 (待生成) 🔲
│       └── distilgpt2_int4.dlc      ← INT4 量化产物 (待生成) 🔲
├── 02_scripts/
│   ├── llm_inference.py             ← 推理脚本
│   ├── export_gpt2_onnx.py
│   └── export_gpt2_onnx_v2.py
└── 03_output/
    ├── calib/                        ← 10 条校准数据 (已改为 int32，匹配 DLC 输入)
    ├── calib_f32_backup/             ← 原 float32 校准备份 🆕
    └── convert.log                  ← snpe-onnx-to-dlc 重跑日志 🆕
```

> 🧹 已清理废弃文件（释放 ~2.3G，磁盘 76%→68%）：`distilgpt2_onnx/`(opset14 失败版)、`distilgpt2_sim.onnx`、`distilgpt2_seq32.bin`(没用到)、`int4/`(abandoned)、`qnn/`(未引用)。

---

## 五、踩坑记录（Day 7 新增）

| # | 描述 | 解法 |
|----|------|------|
| 20 | `qairt-quantizer` float_fallback 和 input_list 互斥 | ❌ 放弃此路线 |
| 21 | Gather_6 动态索引 → CPU 后端无法校验（量化 + 推理都死在这） | Python 脚本用 Reshape→Slice→Squeeze 替换 |
| 22 | `snpe-net-run` 和 `qnn-net-run` 参数完全不同 | 统一用 `snpe-net-run --container` |
| 23 | transformers 离线模式下 `local_files_only=True` 才不联网 | 加载本地 tokenizer 目录 |
| 24 | 校准 raw 是 float32，但 DLC 输入是 Int_32 → 量化读数据会读出垃圾值 | `np.float32.astype(np.int32)` 重写 30 个 raw，备份在 `calib_f32_backup/` |
| 25 | `np.fromfile(p).astype(int32).tofile(p)` 就地写回批量时写成全 0 | 改为先写 `/tmp` 再 `shutil.move` 原子覆盖 |
| 26 | ONNX→DLC 转换在 VM 上极慢（10+ 分钟），看似卡死 | 非卡死：x86-on-ARM QEMU 模拟(5-10x) + 转换器单线程；加内存无用，只能等 |
| 27 | Claude Code 拒绝/中断 Bash 工具调用时，只杀本地 ssh，VM 上已启动的远程进程会成孤儿继续跑 | 远程重活要用 nohup 或主动查 `ps`，不能假设「拒绝=没执行」 |
| 28 | `snpe-onnx-to-dlc` 转换 fixed.onnx 首次跑了 17 分钟后进程消失、无 DLC 产出（疑似 OOM 或序列化阶段崩溃，但孤儿进程没留 stderr 无法确认） | 重跑必须 `nohup ... > convert.log 2>&1 &` 落盘日志；监控要盯「父进程」而非 worker PID（多进程，worker 会换） |
| 29 | fix_dynamic_gather.py 第一次用 Reshape→Slice→Squeeze 替换，Reshape shape 参数错误（32≠24576）导致转换失败 | 第二次用 Constant[31] 替换 index，但下游 Reshape_3 的 shape 来自 Add_1(动态) 仍然出错(1≠32) |
| 30 | 第三次修复：Gather 算子保留，只把 index 从动态 Add_1 换为 Constant[0..31]（32个值），ONNX→DLC 成功 | 完整理解链路：Flatten[32,768] → Gather_6[i=[0..31]] → Reshape_3 → logits，常量 index 让 CPU 后端能静态推断 shape |
| 31 | LLM INT8 量化 5 次尝试全部 segfault（snpe-dlc-quantize ×3，qairt-quantizer ×2），扩内存到12G也无效 | **SDK bug — 非内存问题**。SNPE v2.22.6 C++ 量化后端对大 Transformer（461MB/1496节点）存在兼容性 bug，RSS 仅 500MB 时即崩溃。CNN 模型（YOLO 29MB / Inception 92MB）量化正常。LLM 量化需真机 HTP 原生工具链或新版 SDK。 |

---

## 六、GPT-2 INT8 量化失败完整分析（2026-07-02）

### 尝试记录

| # | 工具 | 配置 | 校准数据 | VM 内存 | 结果 |
|---|------|------|---------|---------|------|
| 1 | `snpe-dlc-quantize` | 默认 | 10 条 int32 | 7.7G | segfault @ "Quantized parameters" |
| 2 | `snpe-dlc-quantize` | 默认 | 1 条（减内存） | 7.7G | segfault @ 同一位置 |
| 3 | `qairt-quantizer` | `--weights_bitwidth 8 --act_bitwidth 8` | 10 条 | 7.7G | segfault |
| 4 | `qairt-quantizer` | `--use_native_input_files` | 10 条 | 7.7G | segfault |
| 5 | `snpe-dlc-quantize` | 默认 | 10 条 | **12G** | segfault @ 同一位置 |

### 排除的错误原因

| 怀疑 | 验证 | 结论 |
|------|------|------|
| 内存不足 | 扩到 12G 仍 crash，RSS 最高才 509MB | ❌ 不是 |
| 校准数据格式错误 | 验证 int32 值正确（token ids 0~50000, mask=1, pos=0~31） | ❌ 不是 |
| DLC 文件损坏 | `snpe-dlc-info` 验证通过，所有节点正常 | ❌ 不是 |
| 校准数量过多 | 单条数据也 crash | ❌ 不是 |
| 工具本身损坏 | YOLOv5s (29MB) 量化成功 | ❌ 不是 |

### 根因：SNPE v2.22.6 C++ 后端对 Transformer 的兼容性 bug

```
所有 segfault 都发生在 "Quantized parameters" 日志之后
→ C++ 后端的量化参数计算阶段崩溃
→ CNN 模型不走相同代码路径（无 Gather/Reshape 链）
→ Transformer 特有算子链触发了未测试的边缘情况
```

| 模型 | 类型 | DLC 大小 | 节点数 | 量化 |
|------|------|---------|--------|------|
| YOLOv5s | CNN | 29MB | ~200 | ✅ 成功 |
| InceptionV3 | CNN | 92MB | ~400 | ✅ 成功 |
| DistilGPT-2 | **Transformer** | **461MB** | **~1496** | ❌ segfault |

### 结论

SNPE v2.22.6（2024年发布）的量化工具链主要为 CNN 优化，LLM Transformer 量化支持不成熟。需在真机（Orion O8G2 / QCS8550）上使用 QNN HTP 原生工具链进行量化。

---

## 七、真机板连接 + 模型传输记录（2026-07-02）

### Orion O8G2 板子规格

| 项目 | 规格 |
|------|------|
| 芯片 | Qualcomm SM8550 (骁龙 8 Gen 2) / Kailua |
| CPU | 8核 ARMv8 (1×X3 + 2×A715 + 2×A710 + 3×A510) |
| RAM | 24 GB DDR5 @ 4224MHz |
| 存储 | 256 GB UFS 4.0 |
| GPU | Adreno 740 |
| HTP | QNN HTP V73 (libQnnHtpV73Skel.so) |
| OS | Android 13 (Linux 5.15.78) |
| 串口 | FTDI FT232R @ 115200 8N1 |
| 网络 | RTL8125 2.5GbE, IP 192.168.100.1 |

### 文件传输方式

```
VM(192.168.64.3) --scp--> Mac(192.168.116.66) --HTTP:8899--> Board(192.168.100.1)
```

板子用 `curl` 从 Mac HTTP Server 下载。

### 板子上的模型文件

```
/data/local/tmp/models/
├── yolov5s.dlc               28MB   (FP32)
├── yolov5s_quantized.dlc    7.1MB   (INT8, VM上量化)
├── inception_v3.dlc          91MB   (FP32)
├── inception_v3_int8.dlc     23MB   (INT8, 今日量化, -74%)
└── distilgpt2_fixed.dlc     460MB   (FP32, 今日转换)
```

### 板子上的 QNN 运行时

```
/vendor/lib64/
├── libQnnCpu.so              ✅ CPU 后端
├── libQnnHtp.so              ✅ HTP 后端
├── libQnnHtpV73Stub.so       ✅ HTP V73 Stub
├── libQnnHtpPrepare.so       ✅ HTP 预处理
├── libQnnSystem.so           ✅ 系统接口
└── libqnnengine.so           ✅ Android ML Engine

/vendor/lib/rfsa/adsp/
└── libQnnHtpV73Skel.so       ✅ HTP FastRPC Skeleton (DSP端)
```

> ⚠️ 板子只有运行时，没有 `snpe-net-run`/`qnn-net-run` CLI 工具。需从 SDK 交叉编译 ARM64 版推理工具，或通过 Android NNAPI 调用 QNN 引擎。
