# Day 2：安装 SNPE

> 端侧 AI 测试（端测）学习 · 第二天
> 目标：装好高通 SNPE/QNN SDK + Python 环境，跑出版本号
> 日期：2026-06-29

---

## 一、原始计划 vs 实际执行

| 步骤 | 原计划 | 实际 |
|------|--------|------|
| 2.1 连接虚拟机 | ssh steven@192.168.64.3 | ✅ 连上（DNS 又挂了，修好） |
| 2.2 安装 QPM3 | 下 deb → dpkg -i 安装 | ❌ 跳过。Server 版无图形界面，QPM3 无法用 |
| 2.3 下载 SNPE | QPM3 图形界面登录下载 | ✅ 改用 wget 直链下载 |
| 2.4 确认路径 | `/opt/qcom/aistack/snpe/` | ✅ 实际路径 `~/qairt/2.22.6.240515`（QNN SDK，兼容 snpe-* 命令） |
| 2.5 配环境变量 | export + source envsetup.sh | ✅ 已配，`snpe-net-run --version` 尚不能运行（GLIBC 不够） |
| 2.6 装 Python3.8 | apt install python3.8-venv | ⏳ 待做 |
| 2.7 装 AI 框架 | TF/ONNX/PyTorch | ⏳ 待做 |
| 2.8 写恢复脚本 | ~/snpe_env.sh | ⏳ 待做 |

---

## 二、已完成步骤（详细记录）

### 2.1 启动虚拟机并连接

```bash
# Mac 终端
ssh steven@192.168.64.3
```

> ⚠️ 每次重启后 DNS 都会失效，需要重新修复（见下方 2.1.1）。

#### 2.1.1 修复 DNS（每次重启必做）

**现象：** `ping 8.8.8.8` 能通，但 `ping mirrors.aliyun.com` 报 `Temporary failure in name resolution`。

**原因：** `/etc/resolv.conf` 是个软链接，指向 systemd-resolved 管理的不存在的文件。重启后被还原。

**修复：**
```bash
sudo rm -f /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" | sudo tee /etc/resolv.conf
sudo systemctl disable systemd-resolved --now
```

**验证：**
```bash
ping -c 1 mirrors.aliyun.com
# 看到 64 bytes from ... 即可
```

### 2.2 跳过 QPM3，改用 wget 直链下载

**原因：** Ubuntu Server 无图形界面，QPM3 是图形工具，无法使用。

**替代方案：** 高通提供了 SNPE/QNN SDK 的公开直链下载（无需登录），直接 wget：

```bash
cd ~
wget https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip
```

### 2.3 解压 SDK

```bash
# 装 unzip（如果没装）
sudo apt install -y unzip

# 解压
unzip v2.22.6.240515.zip
# 解压到 ~/qairt/2.22.6.240515/
```

**目录结构：**
```
~/qairt/2.22.6.240515/
├── benchmarks/
├── bin/
│   ├── x86_64-linux-clang/   ← 主机推理二进制（我们的平台）
│   │   ├── snpe-net-run       ← 兼容 SNPE 命令
│   │   ├── snpe-onnx-to-dlc
│   │   ├── snpe-dlc-quantize
│   │   ├── snpe-diagview
│   │   ├── snpe-throughput-net-run
│   │   ├── qnn-net-run        ← QNN 新命令（也包含同样功能的）
│   │   ├── qnn-onnx-converter
│   │   └── ...（更多工具）
│   ├── envsetup.sh
│   └── check-python-dependency
├── examples/
├── include/
├── lib/
│   └── x86_64-linux-clang/    ← SDK 库文件
├── docs/
└── sdk.yaml
```

> 📌 关键发现：这个 SDK **同时包含 snpe-* 和 qnn-* 命令**。QNN 是 SNPE 的后继者，但老命令名仍然保留，文档④中的 SNPE 命令全部可用。路径前缀从 `snpe-` 不变，直接兼容。

### 2.4 配置环境变量

```bash
export SNPE_ROOT=~/qairt/2.22.6.240515
source ${SNPE_ROOT}/bin/envsetup.sh
```

输出：
```
[INFO] AISW SDK environment set
[INFO] QNN_SDK_ROOT: /home/steven/qairt/2.22.6.240515
[INFO] SNPE_ROOT: /home/steven/qairt/2.22.6.240515
```

✅ 环境配好，PATH 已加入 `~/qairt/2.22.6.240515/bin/x86_64-linux-clang/`。

### 2.5 尝试运行 snpe-net-run（踩坑中）

```bash
snpe-net-run --version
```

#### 坑 1：libunwind.so.1 缺失

**报错：**
```
snpe-net-run: error while loading shared libraries: libunwind.so.1: cannot open shared object file
```

**排查：**
```bash
find /usr/lib -name "libunwind*"
# 系统只有 libunwind.so.8，没有 .so.1
```

**绕过：**
```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libunwind.so.8 /usr/lib/x86_64-linux-gnu/libunwind.so.1
sudo ldconfig
```

#### 坑 2：GLIBC 版本过低（核心问题）

**报错：** 一大串 `GLIBC_X.XX not found`：
```
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.33' not found
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
/lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.35' not found
```

**原因：** Ubuntu 20.04 自带 GLIBC 2.31，但 QNN SDK v2.22.6 **要求 GLIBC ≥ 2.32**（实际需要 2.32~2.35）。

**解决方向：** 升级 Ubuntu 20.04 → 22.04（GLIBC 会升到 2.35）。

---

## 三、Ubuntu 升级（已完成 ✅）

> 已于 2026-06-29 深夜完成升级。

```bash
sudo do-release-upgrade
```

当时弹出的提示及应对（供参考）：

| 提示 | 选择 |
|------|------|
| "Continue running under SSH?" | **y** |
| 是否继续升级 | **y** |
| 配置文件冲突（resolv.conf 等） | **keep the local version** |
| 是否移除过时包 | **y** |
| 是否重启 | **y** |

---

## 四、升级后验证与环境搭建（已全部完成 ✅）

### 4.1 验证系统版本 ✅

```
Ubuntu 22.04.5 LTS (jammy)
GLIBC 2.35
```

### 4.2 DNS ✅

> ⚠️ 升级后 DNS 被还原，已修复。

### 4.3 SNPE 版本验证 ✅

```
SNPE v2.22.6.240515184619_92920
```

### 4.4 Python 虚拟环境 ✅

- 使用 Ubuntu 22.04 自带 **Python 3.10.12**（非 3.8）
- venv 路径：`~/tools/venv/snpe/`

### 4.5 AI 框架 ✅

| 包 | 版本 |
|---|------|
| tensorflow | 2.10.1 |
| onnx | 1.13.0 |
| torch | 2.12.1+cpu |
| torchvision | 0.27.1+cpu |

> ⚠️ **ONNX 版本踩坑**：最初按文档装 `onnx==1.11.0`，该版本无 Python 3.10 的预编译 wheel，pip 会下载源码现场编译，链接系统静态 libprotobuf.a 时因缺少 -fPIC 而失败。**改成 `onnx==1.13.0` 直接装预编译 wheel，秒过。** SDK 的 `snpe-onnx-to-dlc` 是独立二进制，不依赖 Python onnx 包版本，1.13.0 完全可用。

### 4.6 SDK Python 依赖检查 ✅

`check-python-dependency` 输出 30 个包全部已装，9 个版本略高于推荐版本（WARNING 级别），不影响模型转换和推理。

### 4.7 恢复脚本 ✅

已写入 `~/snpe_env.sh`，每次登录只需：
```bash
source ~/snpe_env.sh
```

### 4.8 备份快照

```bash
# Mac 终端执行（虚拟机需关机）
cp -r ~/Library/Containers/com.utmapp.UTM/Data/Documents/ubuntu-snpe.utm \
      ~/Documents/端测/ubuntu-snpe-day2-clean.utm
```

> 📌 UTM 虚拟机实际路径：`~/Library/Containers/com.utmapp.UTM/Data/Documents/ubuntu-snpe.utm`
> 已有一个 Day 1 备份：`~/Documents/端测/ubuntu-snap-backup-day1.utm`

---

## 五、下次开机必备指令

> ⚠️ 虚拟机重启后，以下事项必须重新做！

### 第一步：修复 DNS（100% 会被还原）

```bash
sudo rm -f /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" | sudo tee /etc/resolv.conf
sudo systemctl disable systemd-resolved --now
ping -c 1 mirrors.aliyun.com   # 验证
```

### 第二步：一键恢复 SNPE 环境

```bash
source ~/snpe_env.sh
```

> 这条命令会同时激活 Python 虚拟环境 + 设置 SNPE 环境变量。后续所有操作都在这个环境下进行。

### 第三步（可选）：验证一切就绪

```bash
snpe-net-run --version   # SNPE v2.22.6.240515184619_92920
python3 --version        # Python 3.10.12
pip list | grep -iE "tensorflow|onnx|torch"
```

---

## 六、Day 2 完成清单

| # | 任务 | 状态 |
|---|------|------|
| 1 | 启动虚拟机，ssh 连接 | ✅ |
| 2 | 下载 QNN/SNPE SDK | ✅ 直链 wget |
| 3 | 解压到 `~/qairt/2.22.6.240515` | ✅ |
| 4 | 配好环境变量 | ✅ |
| 5 | 修复 libunwind.so.1 | ✅ 软链接绕过 |
| 6 | 升级 Ubuntu 20.04 → 22.04 | ✅ |
| 7 | `snpe-net-run --version` 出版本号 | ✅ SNPE v2.22.6 |
| 8 | Python 虚拟环境建好 | ✅ Python 3.10, ~/tools/venv/snpe/ |
| 9 | TensorFlow / ONNX / PyTorch 装好 | ✅ TF 2.10.1 / ONNX 1.13.0 / Torch 2.12.1 |
| 10 | 环境恢复脚本写好 | ✅ ~/snpe_env.sh |
| 11 | 拍 Day 2 备份快照 | ✅ ubuntu-snpe-day2-clean.utm |

---

## 七、已执行的命令速查

### 下载 & 解压
```bash
cd ~
wget https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip
sudo apt install -y unzip
unzip v2.22.6.240515.zip
# → ~/qairt/2.22.6.240515/
```

### 环境变量
```bash
export SNPE_ROOT=~/qairt/2.22.6.240515
source ${SNPE_ROOT}/bin/envsetup.sh
```

### 依赖修复
```bash
# libunwind
sudo ln -s /usr/lib/x86_64-linux-gnu/libunwind.so.8 /usr/lib/x86_64-linux-gnu/libunwind.so.1
sudo ldconfig

# 升级系统（已执行，无需再跑）
sudo apt update && sudo apt install -y update-manager-core
sudo apt upgrade -y
sudo do-release-upgrade
```

### Python 环境（已执行，记录备用）
```bash
sudo apt install -y python3.10-venv
python3 -m venv ~/tools/venv/snpe/
source ~/tools/venv/snpe/bin/activate
pip install --upgrade pip
pip install tensorflow==2.10.1
pip install onnx==1.13.0          # 非 1.11.0，见踩坑 #6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### DNS 修复（每次重启必做）
```bash
sudo rm -f /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" | sudo tee /etc/resolv.conf
sudo systemctl disable systemd-resolved --now
```

---

## 八、踩坑记录

| # | 坑 | 现象 | 解决 |
|---|-----|------|------|
| 1 | **DNS 重启后反复失效** | `/etc/resolv.conf` 软链接每次重启被还原，域名解析失败 | 每次重启后手动重建 resolv.conf + disable systemd-resolved |
| 2 | **QPM3 无法在 Server 版使用** | 图形界面工具，无桌面环境打不开 | wget 直链替代，跳过 QPM3 |
| 3 | **下载下来的是 QNN 而非 SNPE** | 以为白下了 | QNN 同时包含 snpe-* 命令，完全兼容 SNPE 旧用法 |
| 4 | **libunwind.so.1 缺失** | 系统只有 .so.8，SDK 要 .so.1 | 软链接从 .so.8 → .so.1 |
| 5 | **GLIBC 版本过低** | Ubuntu 20.04 自带 2.31，SDK 要 2.32~2.35 | 升级到 22.04 |
| 6 | **ONNX 1.11.0 源码编译失败** | 无 Python 3.10 预编译 wheel，pip 源码编译链接静态 libprotobuf.a 缺 -fPIC 报错 | 改用 onnx==1.13.0（有预编译 wheel），SDK 的 snpe-onnx-to-dlc 是独立二进制不依赖 Python onnx 版本 |

---

## 九、Day 2 最终环境快照

```
操作系统：Ubuntu 22.04.5 LTS (jammy) x86_64
GLIBC：   2.35
Python：  3.10.12 (venv: ~/tools/venv/snpe/)
SNPE：    v2.22.6.240515184619_92920 (@ ~/qairt/2.22.6.240515)

关键包：
  tensorflow         2.10.1
  onnx               1.13.0
  torch              2.12.1+cpu
  torchvision        0.27.1+cpu
  opencv-python      4.5.4.58
  numpy              1.26.4

恢复脚本：~/snpe_env.sh
```
