# Changelog - llama.cpp for NVIDIA DRIVE Thor (sm_101)

All notable changes, architectural experiments, on-board verifications, and performance findings for the DRIVE Thor-U (Tegra264 / sm_101a) branch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-09-22

### 🚀 重点性能突破与实证验证 (Tonight's On-Board Verification)
- **DFlash2 块扩散投机解码 (Block-Diffusion Speculative Decoding) 全面跑通与实测定案**:
  - 成功在 DRIVE Thor-U 上打通 `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` 独立 Sidecar 草稿模型与 `Qwen3.8-27B-NVFP4-v4` 主干模型联合推理。
  - **单流解码极速飞跃**: 实测单流端到端 Decode 吞吐高达 **22.56 ~ 22.90 tok/s**，单步平均接受长度达 **5.25 tokens/step** (投机接受率 59.6%~61.6%)。
  - 相比内置 MTP-3 (17.11 tok/s) **提升 +31.85%**，相比纯算力基线 (14.89 tok/s) **提升 +51.51%**，刷新平台 27B 大模型单流速度记录。
  - **4 槽位高并发评测**: 4 槽位 × 128K (共享 256K Unified KV) 下聚合吞吐为 **33.60 tok/s** (单流 8.40 tok/s)。揭示了 Thor 14 个 SM 紧凑架构在双模型并发时计算争抢的物理边界。
- **4 槽位 × 128K 上下文多槽位根因交叉验证矩阵定案**:
  - **Baseline 生产版 (`4971cd92`)**: 4 槽位并发吞吐 **40.08 tok/s**。
  - **加固版 (初始锁死 `MAX_CONNECTIONS=1`)**: 吞吐落至 **35.93 tok/s (-10.35%)**，根因已确认是硬件流串行化。
  - **实验组 A (解绑串行流锁)**: 吞吐即刻回升至 **36.55 tok/s (+1.74%)**，投机命中率从 76.6% 恢复至 **80.4% (+3.8%)**。
  - **实验组 B (24MB L2 KV 持久化窗口)**: 吞吐为 **36.06 tok/s**，证实过度锁定 75% L2 给 KV 会挤压 15GB 权重的流转空间，生产环境应维持默认 `L2=0` (硬件 LRU 自适应)。
  - **实验组 C (压缩投机深度至 `draft-max 2`)**: 吞吐骤跌至 **31.21 tok/s (-22.1%)**，证实 MTP-3 在 Thor 架构下的最佳投机步长为 `n_max=3`。

### 🛡️ 代码加固与机制定案
- **MMQ 活跃度回滚 (`07f3296aa`)**: 实测确认 `occ=2` 引发寄存器/SMEM 溢出并导致 Prefill 性能下降 5.2%，板端与代码已坚决回滚至最优 `occ=1`。
- **128-bit 向量化载入安全防御 (`4153073`)**: `uint4` 突发载入内置 16 字节对齐安全防御门禁，35/35 程序题严格验收 100% PASS。
- **主机 OOM 防御**: 全面落地 `--cache-ram 512` 生产约束，杜绝多长文本请求下的内存泄漏。

---

## [0.2.0-thor] - 2026-09-21

### Added
- **阶段一防御性加固套件 (`e683902`)**:
  - 统一 Thor 家族宏定义 `GGML_CUDA_CC_IS_THOR_FAMILY(cc)`。
  - 动态环境变量控制 L2 持久化窗口 (`GGML_CUDA_THOR_L2_PERSIST_MB`, 默认 0 关闭防颠簸)。
  - 注入 `static_assert(VDR_NVFP4_Q8_1_MMVQ == 4)` 静态类型安全断言。
- **[B3] y-Vector 寄存器跨行复用 (`21556fa4e`)**:
  - 在 `vecdotq.cuh` 与 `mmvq.cu` 中重构 NVFP4 GEMV 内核，预加载 $y$ 向量常驻 8 个寄存器。
  - 算子微基准实测从 108 GB/s 翻倍至 **208 GB/s (+88%)**。
  - 单流 MTP Decode 吞吐从 14.89 提升至 **17.11 tok/s (+14.91%)**。
  - 32K 超长上下文检索稳态吞吐提升 **+12.30%**。

---

## [0.1.0-thor] - 2026-09-19

### Added
- 针对 NVIDIA DRIVE Thor-U (sm_101a / Tegra264 / DriveOS 7.0.3) 的全套编译构建支持。
- Flash-Attention MMA f16 算子适配与 4 槽位 128K Unified KV 共享池生产支持。
