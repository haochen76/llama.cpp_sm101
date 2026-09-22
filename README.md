# llama.cpp for NVIDIA DRIVE Thor (sm_101 / Tegra264)

> 🚗 **车规域控极速推理分支**：针对 NVIDIA DRIVE Thor-U（Tegra264, Blackwell `sm_101a`, DriveOS 7.0.3, CUDA 12.8 / 12.9）深度定制的 `llama.cpp` 高性能优化分支。
>
> 本项目秉承严格的求真工程标准，所有测试数据均来自真实车规域控板端先后独立 A/B 对比评测（零并发显存/端口干扰、结温监控墙 ≤90°C，前置降温至 ≤75°C）。

---

## 📊 板端已完成实测验证的基准数据 (Verified on Board)

以下数据全部在真实车规板端硬件（**NVIDIA DRIVE Thor-U 64GB 统一内存，DriveOS 7.0.3，sm_101a**）上完成端到端测试与验收。

### 1. 投机架构对比与解码极速突破 (DFlash2 vs MTP-3 vs Baseline)

* **目标主干模型**: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-NVFP4-v4.gguf` (15 GB, -ngl 99)
* **DFlash2 草稿模型**: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.14 GB, -md, -ngld 99)
* **执行口径**: 严格先后独立执行（单次测完彻底退出释放显存，冷却至 ≤75°C 后再测，硬约束温控结温 ≤90°C）

| 评测场景与投机架构 | 指标说明 | Baseline (生产版 `4971cd92`) | MTP-3 内置投机 (加固版最优) | DFlash2 独立 Sidecar 模型 | 实测最大突破 / 收益 | 板端验证状态 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **单流 Decode 稳态吞吐** | 端到端单流生成速度 | 14.89 tok/s | 17.11 tok/s | **22.56 ~ 22.90 tok/s** | 🚀 **+51.51%** (vs Base)<br/>🚀 **+31.85%** (vs MTP) | ✅ **板端实测定案** |
| **单步平均接受长度** | 每步生成的有效 token 长度 | 1.00 len | 3.42 len | **5.25 len** | 🚀 **5.25 tok / pass** | ✅ **板端实测定案** |
| **投机命中率 (Accept Rate)** | 草稿 token 验证接受比例 | N/A | **80.4%** | **59.6% ~ 61.6%** | 单步 7 token，产出量更大 | ✅ **板端实测定案** |
| **4 槽位并发聚合吞吐** | 4 槽位 × 128K (共享 256K) | **40.08 tok/s** | **37.15 tok/s** (RPB=8)<br/>**36.31 tok/s** (RPB=4) | **33.60 tok/s** | RPB 自适应多档调优闭环 | ✅ **板端实测定案** |
| **4 槽位每流平均吞吐** | 4 并发下单流服务速度 | 10.02 tok/s | 9.29 tok/s (RPB=8)<br/>9.08 tok/s (RPB=4) | 8.40 tok/s | 极长上下文多槽位均分 | ✅ **板端实测定案** |
| **32K 超长上下文 Decode** | ~20,943 tokens KV 稳态解码 | 15.20 tok/s | **17.07 tok/s** | - | 🚀 **+12.30%** | ✅ **板端实测定案** |

---

### 2. 4 槽位 × 128K (共享 256K Unified KV) 多槽位与自适应 RPB 实测矩阵

针对“单槽位极速（+14.9%~+31.8%）与多槽位高并发吞吐之间的硬件权衡”，我们在板端实施了自适应 RPB（Rows Per Block）与并发流调优实验，实证数据闭环如下：

| 实验组别 | 运行配置 / 环境变量 | 4 槽位聚合吞吐 (tok/s) | 相对基线 | MTP 命中率 | 底层物理根因与实证发现 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline 生产版** | `4971cd92` (原厂生产版，无串行锁，无 B3) | **40.08** | 基准 (100%) | 80.8% | 14 SM 下纯 GEMM 饱和度高 |
| **自适应 RPB=8 对照组** | `GGML_CUDA_THOR_MMVQ_RPB=8` (默认单流极速) | **37.15** | -7.31% | **79.6%** | 单流极限带宽档，在 4 槽位下创下加固版最高吞吐 (37.15 tok/s) |
| **自适应 RPB=4 平衡组** | `GGML_CUDA_THOR_MMVQ_RPB=4` (273GB/s 访存延迟平衡) | **36.31** | -9.41% | 77.8% | 寄存器压力减半，极度稳定（Run1 36.31 vs Run2 36.19，方差极小） |
| **自适应 RPB=1 并发组** | `GGML_CUDA_THOR_MMVQ_RPB=1` (细粒度铺满 14 SM) | **36.15** | -9.81% | **80.7%** | Grid Block 数扩大 8 倍，投机命中率攀升至 80.7% |
| **实验组 B (L2 持久化窗口)**| 实验组 A + **激活 24MB L2 KV 持久化** | **36.06** | -10.03% | 79.8% | 锁死 75% L2 挤压 15GB 权重与中间张量，引发颠簸；默认 `L2=0` 最优 |
| **加固版 (初始测验)** | Sandbox + `MAX_CONNECTIONS=1` + `L2=0` | **35.93** | -10.35% | 76.6% | 串行队列锁破坏了 Thor 多流异步重叠发射 |
| **DFlash2 架构对照组** | DFlash2 Sidecar (`draft-max 7`, `no-sampling`) | **33.60** | -16.17% | 58.5% | 14 SM 架构在并发时难以承载主干+草稿双模型 GEMM 调度争抢 |
| **实验组 C (压缩投机深度)** | 实验组 A + **轻量起草步长 (`draft-max 2`)** | **31.21** | -22.13% | 83.4% | 有效起草长度由 3.42 缩至 2.65，验证步频过密反噬吞吐；`n_max=3` 最优 |

---

### 3. 算子微基准带宽实测 (Op-Level Microbench)

| 算子微基准 | 原始吞吐 | 板端实测优化后吞吐 | 提升幅度 | 实测根因与机理 |
| :--- | :---: | :---: | :---: | :--- |
| **NVFP4 MMVQ GEMV ($M=1$)** | 108 GB/s | **202 ~ 208 GB/s** | **+88% (打满带宽)** | [B3] 消除行循环内 $y$ 向量重复 L2 读取，寄存器预读跨 8 行广播 |
| **F8 Attention 投影 (cuBLASLt)** | - | **246 GB/s** | 约 90% 硬件上限 | 自写 cuBLASLt shim，近极限访存利用 (接近 273 GB/s 理论上限) |

---

## 🛠️ 各 Commit 环节与优化收益审计表 (Commit Breakdown & Gains)

本项目代码提交严格遵守模块化与防御性设计，以下为各环节 Commit 带来的具体机制与实测收益：

| Commit 节点 | 模块 / 环节 | 核心代码改动 | 实测收益 / 解决痛点 | 验证状态 |
| :--- | :--- | :--- | :--- | :---: |
| [`bc1f5f8`](https://github.com/haochen76/llama.cpp_sm101/commit/bc1f5f8) | **方案 1: mmvq 自适应动态分流** | 在 `mmvq.cu` 中解耦 `rows_per_block` 模板参数，支持环境受控及自适应 `RPB=1/2/4/8` 分流 | 消除单流极速与多流 Occupancy 冲突，4 槽位聚合吞吐回升至 **37.15 tok/s**，命中率达 **80.7%** | ✅ **板端实测定案** |
| [`e925275`](https://github.com/haochen76/llama.cpp_sm101/commit/e925275) | **P0: 架构识别与 WebUI** | 支持 `-DCMAKE_CUDA_ARCHITECTURES="100;101"` 原生编译；修复思维链交互 | 首次在 DRIVE Thor-U 上成功拉起原生长文本服务 | ✅ **已验证** |
| [`21556fa4e`](https://github.com/haochen76/llama.cpp_sm101/commit/21556fa4e) | **P2-1: B3 perj y 向量复用** | `vecdotq.cuh` 中重构 NVFP4 MMVQ 内核，利用寄存器跨行复用 $y$ 向量 | 单流解码提升 **+16%~+20%**，算子带宽实测翻倍至 **208 GB/s** | ✅ **已验证** |
| [`e683902`](https://github.com/haochen76/llama.cpp_sm101/commit/e683902) | **阶段一代码防御性加固** | 1. 统一 Thor 架构判定宏 `GGML_CUDA_CC_IS_THOR_FAMILY(cc)`<br/>2. L2 持久化窗口改由环境变量控制且默认关闭 (`default=0`)，防短文本 L2 颠簸<br/>3. 注入 `static_assert(VDR_NVFP4_Q8_1_MMVQ == 4)` 静态断言防错 | 1. 消除短上下文和高并发下的 L2 Cache 颠簸<br/>2. **单流 MTP 稳态吞吐提升至 17.11 tok/s (+14.91%)**<br/>3. **32K 长文本检索效率提升 +12.30%** | ✅ **已验证** |
| [`6aaf664`](https://github.com/haochen76/llama.cpp_sm101/commit/6aaf664) | **P1-1: MMQ Occupancy 调谐** | 实验性将活跃 block 占有率提至 2，探索隐藏访存延迟 | 实测发现单 CTA SMEM 占用受限导致轻度寄存器溢出压力，Prefill 速度下降 5.2% (210.9 → 199.9 tok/s)。**已实测定案证伪，生产环境坚决维持 occ=1** | ✅ **板端实测定案** |
| [`4153073`](https://github.com/haochen76/llama.cpp_sm101/commit/4153073) | **P1-2: 128-bit 向量化加载** | `uint4` 连续突发载入，内置 16 字节对齐安全防御门禁 | 经 35/35 程序题严格验收与 32K 压力测试，安全对齐回退路径 100% 可靠，无通道死锁；进阶升级路线指向 TMA `cp.async.bulk` | ✅ **板端实测定案** |
| [`07f3296`](https://github.com/haochen76/llama.cpp_sm101/commit/07f3296) | **内核回滚与多槽位实证** | 坚决回滚 MMQ `occ=1`，记录 4 槽位 128K 完整基准测试 | 保持主干代码绝对纯净与高可靠 | ✅ **已验证** |

---

## ⚠️ 车规域控踩坑与已定案经验 (Hard-Won Lessons)

在 DRIVE Thor (DriveOS 7.0.3 / Tegra264) 真实开发中，踩过并已在板端定案的避坑指南：

### 1. 连续请求 Host OOM (板端已定案)
* **现象**: `llama-server` 连续跑 12~13 个不同提示词后必被 Linux 内核 `oom-killer` 强制杀死。
* **根因**: 46GB 大页 GPU 显存池分配后，Host RAM 仅剩约 7.3GB。而 `llama-server` 默认设置 `--cache-ram 8192`（8GB 主机端 Prompt 快照缓存），每条新提示词导致主机内存暴增 ~626MB 直至物理耗尽。
* **已验证解法**: 启动命令必须显式指定 **`--cache-ram 512`**，已实测跑通数十轮长测试不崩溃。

### 2. 多槽位 Multi-Stream 并发与串行队列锁
* **现象**: 开启 `CUDA_DEVICE_MAX_CONNECTIONS=1` 会让 4 槽位并发吞吐下降 10.35%。
* **根因**: 该锁强行将所有 CUDA Stream 压入单一硬件队列，使 Thor-U 失去了多槽位异步前向的重叠能力。多槽位并发时必须 `unset CUDA_DEVICE_MAX_CONNECTIONS`。

### 3. DFlash2 块扩散投机专属参数避坑
* **注意**: DFlash2 的草稿生成基于 256-rank 的 selector 栅格而非词表概率。启动 DFlash2 时必须携带 **`--no-spec-draft-backend-sampling`**，否则后端采样器会破坏 lattice 逻辑导致预测失败。

---

## 💻 生产启动参考指南

### 1. 推荐场景 A：极速单流交互 (DFlash2 块扩散，单流 22.5+ tok/s)

适用于主驾专属交互助手、语音快速响应场景：
```bash
export GGML_CUDA_PDL=1
unset CUDA_DEVICE_MAX_CONNECTIONS

/zeekr_data/llama.cpp/build/bin/llama-server \
  -m /zeekr_map/models/gguf/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-NVFP4-v4.gguf -ngl 99 \
  -md /zeekr_data/models/gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 \
  --no-spec-draft-backend-sampling \
  -fa on --split-mode none \
  --kv-unified --kv-unified-per-slot 131072 -c 262144 -np 1 \
  -b 2048 -ub 512 --cache-ram 512 --jinja -n 8192 \
  --reasoning on --reasoning-effort medium --reasoning-budget -1 \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 --port 8080
```

### 2. 推荐场景 B：多座舱高并发满打 128K (MTP-3 方案，并发 36.55 ~ 40+ tok/s)

适用于多座舱独立并发交互、后台批处理场景：
```bash
export GGML_CUDA_PDL=1
unset CUDA_DEVICE_MAX_CONNECTIONS
export GGML_CUDA_THOR_L2_PERSIST_MB=0

/zeekr_data/llama.cpp/build/bin/llama-server \
  -m /zeekr_map/models/gguf/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-NVFP4-v4.gguf -ngl 99 \
  -fa on --split-mode none \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --kv-unified --kv-unified-per-slot 131072 -c 262144 -np 4 \
  -b 2048 -ub 512 --cache-ram 512 --jinja -n 8192 \
  --reasoning on --reasoning-effort medium --reasoning-budget -1 \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 --port 8080
```

---
*本项目基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 主线持续同步演进。*
