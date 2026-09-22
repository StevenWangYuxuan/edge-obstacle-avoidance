# Day 1：搭建 Ubuntu 环境

> 端侧 AI 测试（端测）学习 · 第一天
> 目标：在 Mac 上用 UTM 跑起一个 Ubuntu 20.04 虚拟机，并从 Mac 终端 ssh 进去
> 日期：2026-06-29

---

## 一、今日目标与背景

端测的核心工具 **SNPE**（高通神经网络处理引擎）只提供 **Linux x86_64** 版本，没有 Mac 版。
我的电脑是 **MacBook Pro / Apple M5 Pro / 24GB / macOS 26.5**（ARM 架构），所以需要用 UTM 虚拟机 **模拟** 一个 x86_64 的 Ubuntu 环境来跑 SNPE。

关键认知：SNPE 流程里"模型转换 → 量化 → 主机推理"这几步**不需要真实硬件板子**，在 Ubuntu 虚拟机上就能跑通。只有最后"推到 HTP/DSP 真机推理"才需要美格模组。所以先在虚拟机上学 70% 的流程。

---

## 二、今日完成清单

- [x] 安装 UTM（Mac 虚拟机软件）
- [x] 下载 Ubuntu 20.04 Server 镜像
- [x] 在 UTM 创建 x86_64 虚拟机 `ubuntu-snpe`
- [x] 安装 Ubuntu 20.04 系统
- [x] 修复 DNS，让虚拟机能联网
- [x] 从 Mac 终端 ssh 进虚拟机
- [x] 安装基础开发工具
- [x] 用 Finder 复制 `.utm` 文件做备份（等同快照）
- [x] 注册高通账号（Day 2 下载 SNPE 用）

---

## 三、虚拟机配置参数

| 配置项 | 值 |
|--------|-----|
| 虚拟机名称 | `ubuntu-snpe` |
| 架构 | x86_64 |
| 机器类型 | pc-q35-7.2（Q35 + iCH9，UEFI 引导） |
| CPU | 4 核 |
| 内存 | 8192 MB |
| 硬盘 | 40 GB（稀疏磁盘，按需分配） |
| 系统 | Ubuntu 20.04.6 Server |
| 用户名 | `steven` |
| 主机名 | `snpe` |
| 虚拟机 IP | `192.168.64.3`（DHCP 动态，可能变化） |

---

## 四、关键命令清单

### 4.1 在 Ubuntu 虚拟机内执行的命令

#### 查看网络和 IP
```bash
ip addr show
```
> 找 `enp0s1` 网卡下 `inet` 后面的地址（本次是 `192.168.64.3`）。

#### 测试联网
```bash
ping -c 3 8.8.8.8
```

#### 测试 DNS 解析
```bash
ping -c 3 mirrors.aliyun.com
```

#### 确认 SSH 服务状态
```bash
sudo systemctl status ssh
```
> 按 `q` 退出查看。若未安装：`sudo apt update && sudo apt install -y openssh-server`

#### 查看当前 DNS 配置
```bash
cat /etc/resolv.conf
```

---

### 4.2 修复 DNS（本次踩的坑，重点记录）

Ubuntu 默认 `resolv.conf` 指向本地代理 `127.0.0.53`（systemd-resolved），无法解析外部域名，导致 `apt` 报 `failed to fetch`。修复步骤：

```bash
# 1. 删除旧的 resolv.conf（可能是软链接）
sudo rm -f /etc/resolv.conf

# 2. 写入真实 DNS 服务器
echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" | sudo tee /etc/resolv.conf

# 3. 关掉 systemd-resolved，防止重启后配置被改回
sudo systemctl disable systemd-resolved --now

# 4. 验证 DNS 是否生效
ping -c 3 mirrors.aliyun.com
```

---

### 4.3 换国内软件源（加速 apt 下载）

默认 `archive.ubuntu.com` 在国内慢，换成阿里云镜像：

```bash
# 1. 备份原源
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 2. 替换为阿里云镜像
sudo sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list
sudo sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list
```

---

### 4.4 安装基础开发工具

```bash
sudo apt update && sudo apt install -y vim curl wget build-essential
```

> - `vim`：文本编辑器
> - `curl` / `wget`：下载工具
> - `build-essential`：提供 gcc / make 等编译工具，装 SNPE 依赖会用到

---

### 4.5 关机（拍快照/备份前）

```bash
sudo shutdown now
```

---

### 4.6 在 Mac 终端执行的命令

#### 从 Mac ssh 进虚拟机
```bash
ssh steven@192.168.64.3
```
> - 首次连接问 `Are you sure...` → 输 `yes`
> - 输密码时屏幕不显示任何字符，正常现象，打完回车
> - 若 IP 变了（DHCP），先在 UTM 窗口登录，`ip addr show` 查新 IP

---

## 五、踩过的坑与解决办法

| 坑 | 现象 | 解决 |
|----|------|------|
| 选 Emulate 还是 Virtualize | Apple Silicon 跑 x86 必须 Emulate（Virtualize 只能跑同架构 ARM 系统） | 选 Emulate |
| 机器类型版本号太多 | Q35 有一堆 `pc-q35-2.x`~`7.2`，部分 deprecated | 选数字最大的 `pc-q35-7.2` |
| 装完重启报 `failed unmounting /cdrom` | iso 光盘还挂着，卸载失败（无害提示） | 进 Settings → Drives 删掉挂 iso 的光盘驱动器，从硬盘引导 |
| 启动卡在 `reached target cloud-init target` | cloud-init 等网络元数据超时 | 按几下回车刷新控制台，登录提示就出来了 |
| `apt` 报 `failed to fetch` | DNS 解析失败（resolv.conf 指向 127.0.0.53） | 见 4.2，重建 resolv.conf 写真实 DNS |
| 软件源下载慢 | 默认源在国内访问慢 | 见 4.3，换阿里云镜像 |
| 找不到快照入口 | UTM 菜单的「虚拟机」项需选中 VM 才出现 | 用 Finder 复制 `.utm` 文件做备份，效果一样 |
| **DNS 重启后反复失效** | 虚拟机每次重启，`/etc/resolv.conf` 软链接指向 `../run/systemd/resolve/stub-resolv.conf` 又被还原，DNS 再次解析失败 | 重启后重新执行 DNS 修复：`sudo rm /etc/resolv.conf && echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" \| sudo tee /etc/resolv.conf`，然后 `sudo systemctl disable systemd-resolved --now` |
| **GLIBC 版本过低** | QNN/SNPE SDK v2.22.6 要求 GLIBC 2.32~2.35，Ubuntu 20.04 自带 2.31，`snpe-net-run` 启动报 `GLIBC_X.XX not found` | 升级 Ubuntu 20.04 → 22.04：先 `sudo apt upgrade -y`，再 `sudo do-release-upgrade`。GLIBC 会升到 2.35 |

---

## 六、虚拟机日常使用方式

### 下次连接流程
1. 打开 UTM → 选中 `ubuntu-snpe` → 点 ▶️ 启动
2. 等 VM 启动到登录界面（不需要在 UTM 窗口登录）
3. Mac 终端敲：`ssh steven@192.168.64.3`
4. 连不上 → IP 变了 → UTM 窗口登录，`ip addr show` 查新 IP

### 每次重启后必做（DNS 修复）
```bash
# 如果 ssh 进去后 ping 域名不通，敲这两条：
sudo rm -f /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 114.114.114.114" | sudo tee /etc/resolv.conf
sudo systemctl disable systemd-resolved --now
```

### 备份恢复（土办法）
- 备份：Finder 里复制 `ubuntu-snpe.utm` 文件到别处
- 恢复：把备份的 `.utm` 拖回 UTM，或替换掉坏的那个

### 注意事项
- 虚拟机硬盘是稀疏磁盘，64GB 是上限，实际按需占用
- IP 是 DHCP 动态分配，重启可能变化（如需固定 IP 可改 netplan 配置，本次未做）
- 装环境前先备份，搞坏了能一键还原
- ⚠️ DNS 在虚拟机重启后会还原失效，每次重启后可能需要重新修复 DNS

---

## 七、Day 2 实际进展：安装 SNPE（2026-06-29）

> 预告阶段的 QPM3（图形界面）方案因 Ubuntu Server 无桌面而不可用。
> 实际采用了 wget 直链下载。

1. ✅ 启动虚拟机，ssh 进去（DNS 果然又挂了，修好）
2. ✅ 跳过 QPM3 → 用 wget 直链下载 QNN/SNPE SDK v2.22.6
3. ✅ 解压到 `~/qairt/2.22.6.240515`（含 snpe-* 和 qnn-* 命令，兼容）
4. ✅ 配好环境变量：`export SNPE_ROOT=~/qairt/2.22.6.240515 && source ${SNPE_ROOT}/bin/envsetup.sh`
5. ⏳ 遇到 GLIBC 2.32~2.35 不满足 → 升级 Ubuntu 20.04 → 22.04（进行中）
6. ⏳ 待完成：`snpe-net-run --version` 出版本号
7. ⏳ 待完成：装 Python3.8 虚拟环境 + TensorFlow/ONNX/PyTorch

> 参考文档：《基于高通SNPE的深度学习推理与应用实践 V1.1》第 2 章
> SNPE 安装路径预期：`/opt/qcom/aistack/snpe/2.x.x`
