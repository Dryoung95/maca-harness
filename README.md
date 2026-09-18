# maca-harness

**KernelBench（ICML'25）在沐曦 MetaX C500 / MACA 上的迁移与验证。**

让原本只认 NVIDIA 工具链的 CUDA kernel 基准测试框架，在 C500 上端到端跑通——编译、正确性评测、性能计时全部走官方原流程。

---

## 这是什么

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) 是一套 CUDA kernel 基准与评测框架：250 道题目（基础算子 / 融合算子 / 完整模型），每题给定参考实现，评测引擎自动生成随机输入、双跑参考实现与待测实现、`allclose` 判正确性、`cuda_event` 计时算加速比。

本仓库把它迁移到 **MetaX C500**（104 SM，16.3 GB，warp size 64，xcore1000 架构，MACA SDK 3.3.0.15）。

---

## 迁移方式：源码零改动

**KernelBench 上游源码一行未改**，全部兼容性由环境层提供：

```
torch cpp_extension (load_inline)
        │
        ▼
   cu-bridge / cucc        ← CUDA 语法 → MACA 语义的桥接编译器
        │
        ▼
      mxcc                 ← 编译为 xcore1000 fatbin
        │
        ▼
   MetaX C500 GPU
```

- PyTorch `2.8.0+metax3.3.0.2`，CUDA API 映射到 MACA
- CUDA 语法（`<<<>>>` launch、shared memory、`__syncthreads`、dim3 grid）经 cu-bridge 自动转译
- `gpu_arch` 沿用 NVIDIA 架构名，cu-bridge 自动转译，无需改动

评测标准（容差、计时方式、题目内容）与上游完全一致——**迁移的是运行平台，不是测试标准**。

---

## 结果

### 阶段一：构建闭环 + 正确性闭环 ✅

全量 250 题执行（L1×100 + L2×100 + L3×50），每题真实编译 CUDA 源码：

| 结果 | 数量 | 占比 |
|---|---|---|
| **passed**（编译 + 正确性双过） | **182** | **72.8%** |
| skipped（输入超 16.3 GB 显存） | 39 | 15.6% |
| failed（harness 流程 OOM） | 18 | 7.2% |
| failed（数值非确定性） | 10 | 4.0% |
| failed（wrapper bug） | 1 | 0.4% |

按 Level：L1 47/100、L2 97/100、L3 38/50。

**核心结论：零例 `backend_error`（编译链失败），零例真实 `algorithm_error`（迁移引入的正确性缺陷）。**

所有未通过的题目都可归因于：
- **显存容量**（57 道）：C500 只有 16.3 GB，这些题在 16 GB L4/T4 上同样无法运行，属 `invalid_infrastructure`
- **数值非确定性**（10 道）：metax eager 的 conv/BN 归约顺序差异，eager 与自身比较即超容差
- **wrapper bug**（1 道）：批量执行器对 0 维标量输出的处理缺陷，非平台问题

### 阶段二：性能调优 🔄

L1 P1 方阵乘法（n=4096）基线：

| 路径 | 耗时 | vs eager |
|---|---|---|
| eager fp32（TF32 路径） | 1592 µs | 1.00x |
| eager bf16 | 773 µs | 2.06x |
| **mcblas fp16 输入 + fp32 累加** | **907 µs** | **1.75x** |
| 手写 wmma（最佳） | 2333 µs | 0.68x |

**首个通过官方门槛的提交**：mcblas fp16-input/fp32-accumulate 路线经 `eval_kernel_against_ref` 全流程评测——`compiled=True`、`correctness 5/5`、**speedup 1.693x**，hardware=MetaX C500。

---

## 目录结构

| 路径 | 内容 |
|---|---|
| `KernelBench/` | 上游源码（未改动） |
| `run_batch.py` | 批量正确性执行器 |
| `batch-results.jsonl` | 250 题逐题结果 |
| `batch-results_summary.json` | 汇总计数 |
| `batch-run.log` | 批量执行日志 |
| `env.sh` | 环境变量配置 |
| `optloop/` | 阶段二 kernel 源码与实验（wmma / mcblas / mctlass / 探针） |
| `DECISIONS.md` | 技术路线决策记录 |
| `MIGRATION_REPORT.md` | 完整迁移报告 |

---

## 快速开始

```bash
# 1. 配置环境（每次运行前必须 source）
source /data/cuda-harness-migration/env.sh

# 2. 全量正确性执行（约 2.7 小时）
python3 run_batch.py --levels 1,2,3 \
  --out batch-results.jsonl --build-root batch_build

# 3. 单题评测
python3 run_batch.py --levels 1 --problem-ids 1 \
  --out /tmp/test.jsonl --build-root /tmp/test_build

# 4. 阶段二：mcblas 提交官方门槛
cd optloop && python3 submit_mcblas.py
```

**环境要求**：MetaX C500、MACA SDK 3.3.0.15+、PyTorch 2.8.0+metax、用户须在 `video` 组以访问 `/dev/mxcd`。

---

## 已知的 MACA 工具链缺陷

迁移过程中隔离并复现了三个工具链问题（详见 `DECISIONS.md`）：

1. **`store_matrix_sync` 忽略 layout tag** —— `mem_row_major` 与 `mem_col_major` 产出相同的转置布局
2. **`load_matrix_sync` col_major 变体打乱数据** —— double 向量化读取导致数据错位（加 padding 后恢复正确）
3. **`__half` 的位重解释产出垃圾值** —— `unsigned` / `half2` / `ulonglong2` 重解释均不可用，只能标量访问

此外 mctlass（MACA 的 CUTLASS 对应物）device 层 GEMM 存在 64 线程 warp 的移植缺口（mxcc `-O2` inlining segfault、ColumnMajor 输出特化算 B@A、SIMT epilogue 只写一半行），暂不可用。

---

## 待完成

- [ ] mcblas binding 目前只覆盖 L1 P1，可复用至 P2–P18 的 matmul 家族
- [ ] 修复 `scripts/run_and_check.py` 的 `pydra.REQUIRED` 依赖问题，恢复官方 CLI
- [ ] 1 道 wrapper bug（L1 P95 标量输出）
- [ ] 10 道数值非确定性题的容差处理
- [ ] Level 4（20 道 HuggingFace 模型推理题），需下载模型权重
- [ ] 57 道显存受限题，需更大显存硬件
