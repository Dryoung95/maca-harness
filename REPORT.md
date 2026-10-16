# KernelBench → MetaX C500 (MACA) 迁移与性能基线报告

**仓库**：`Dryoung95/maca-harness`（主交付物）、`Dryoung95/kda-on-maca`（工作流参考 + 缺陷库）
**硬件**：MetaX C500，104 SMs，16.3 GB / 64 GB 两种配置
**SDK**：MACA 3.3.0.15，Torch 2.8.0+metax3.3.0.2
**上游基准**：KernelBench (ScalingIntelligence, ICML 2025)，源码零改动

---

## 1. 执行摘要

KernelBench 全部 270 道题已在 MACA 平台完成迁移并通过官方评测门槛建立性能基线。核心结论：

- **正确性**：L1–L3 **249/250**（未打补丁），250/250（D7 守卫下）；L4 **20/20**
- **性能基线**：**270/270** 完成测量，**262** 条有效计时记录
- **性能优化**：**58/82** GEMM/线性链题目跑赢 eager（官方 gate 认定）
- **平台缺陷**：定位、复现并文档化 **D1–D7** 共 7 项 MACA 工具链缺陷，其中 4 项为静默错误返回
- **conv 性能线**：四条路径全部以测量数据证伪，作为否定结果关闭

---

## 2. 正确性结果（官方门槛建）

对齐 KernelBench 官方评测标准：5 次正确性试验全过、fp32 `atol=rtol=1e-4`、eval 模式、seed 42。

| 范围 | 结果 | 数据来源 |
|---|---|---|
| L1–L3（未打补丁） | **249 / 250** | `align-final.jsonl` |
| L1–L3（D7 守卫下） | **250 / 250** | `runs/p76_d7fixed.jsonl` |
| Level 4（HuggingFace 推理） | **20 / 20** | `l4-results.jsonl` |

唯一未修复的平台失败是 **L1 P76**（见 §4 D7）。

### 2.1 对齐官方标准时发现的上游偏差

迁移过程中发现 upstream 本身的两处问题，已在 harness 层修正而未改动上游源码：

1. **`eval.py` 从不调用 `.eval()`** —— 模型默认处于 train 模式，BatchNorm/Dropout 在两个模型实例间的分歧（3e-4..1e-2）会伪装成内核精度错误。修正后 fp16 路径的精度边界从最初记录的 29 题收窄到真实的 **4 题**。
2. **计时循环运行在 `no_grad` 之外**（`eval.py:616-700`）—— causal LM 因此保留 autograd 图，叠加两份模型副本占用 30–40 GB（见 §3.1）。

---

## 3. 性能基线

### 3.1 全量计时扫描

`perf-baseline.jsonl` —— 每题 5 次正确性试验 + 20 次 `cuda_event` 计时，eval 模式，D7 守卫启用。

| Level | 题数 | 有效计时 | 说明 |
|---|---:|---:|---|
| L1 | 100 | 100 | |
| L2 | 100 | 100 | |
| L3 | 50 | 50 | |
| L4 | 20 | **12** | 见下方注 |
| **合计** | **270** | **262** | |

**L4 计时缺口（8/20）**：P1/P3/P4/P8/P10/P17/P18/P19 在计时启用时 OOM。这**不是缺陷**：8 题在 `no_grad` 下的正确性门槛建全部通过（`l4-results.jsonl` 20/20），OOM 纯粹是容量天花板——upstream 计时循环在 `no_grad` 之外运行，保留的 autograd 图加两份模型副本适配 KernelBench 目标的 80 GB 显卡，超出本机 **63.59 GB**。该限制**如实记录而非打补丁**，因为补丁会改变官方测量语义。

### 3.2 GEMM 家族优化成果

mcblas fp16 输入 / fp32 累加路径，全部通过 `eval_kernel_against_ref`（参考：TF32 开启的 eager fp32）：

| 范围 | 题数 | 跑赢 eager |
|---|---:|---:|
| L1 GEMM family | 13 | 11 |
| L2 linear chains | 37 | 34 |
| L3 使用 `nn.Linear` 的网络 | 32 | 13 |
| **合计** | **82** | **58** |

最大加速：L3 ShallowWideMLP **6.87×**、L2 P56 **6.05×**、L1 large-K **2.80×**。

n=4096 方阵乘的验证天花板：

| 路径 | 延迟 | vs eager |
|---|---:|---:|
| eager fp32（TF32 on） | 1592 µs | 1.00× |
| **mcblas fp16-in/fp32-acc** | **907 µs** | **1.75×**（同时过 1e-4 门槛建的唯一路径） |
| mcblas bf16 | 812 µs | 1.96×（精度 3.8e-3，过不了门槛建） |
| 手写 wmma | 2333 µs | 0.68× |

### 3.3 conv 家族：测量过的否定结果

backlog 指明的最后一个性能方向（"卷积网络需要 mcdnn 路径"）已被打开并证伪，四条路径全部有探针数据支持（`docs/conv-family/`）：

| # | 路径 | 证伪依据 |
|---|---|---|
| 1 | fp16 conv | SDK **不存在可用路径**：`f16→f32` 返回 `BAD_PARAM`；`f16→f16` 执行但 maxdiff 5.7e-2（错误，非不精确） |
| 2 | 直调 mcdnn | 与 eager 是**同一计算路径**：唯一过门槛建的组合（`PRECOMP_GEMM`+`DEFAULT_MATH`）maxdiff 恰为 0.000、速度恰为 1.00×——torch 本身就链接同一个 `libmcdnn.so` |
| 3 | FMA 精确 fp32 | **比参考更精确反而过不了门槛建**：参考是 TF32，去除 TF32 近似会被判为误差（4.5e-2） |
| 4 | im2col + fp16 GEMM | 双输：`F.unfold` 单纯内存搬运就花掉整个 eager conv 的 2–3×（P55 的 im2col 矩阵 9.66 GB），精度 3–6e-2 |

**关键洞见**：8 个前向算法中只有 `IMPLICIT_PRECOMP_GEMM` 和 `FFT_TILING` 被实例化，其余 6 个（含 Winograd）在 workspace 查询即返回 `NOT_SUPPORTED`。推进 conv 需要**厂商侧改动**，harness 侧无计可施。**58/82 是性能线最终数字。**

---

## 4. MACA 平台缺陷（D1–D7）

全部孤立、复现并文档化，复现脚本入库 `defects/repro/`：

| # | 缺陷 | 静默? | 规避 |
|---|---|---|---|
| D1 | mctlass device-GEMM epilogue 只写每个 16 行 tile 的一半行 | **是** | 无——该层不可用 |
| D2 | `wmma::store_matrix_sync` 忽略 layout tag | **是** | store 后转置 |
| D3 | `wmma::load_matrix_sync` 对 ≥32 lane 返回垃圾 | **是** | 无——向量化加载不可用 |
| D4 | mxcc `-O2` 内联在 mctlass host 侧构造时 segfault | 否（崩溃） | `-fno-inline` |
| D5 | fp32/TF32 `wmma` fragment 未实例化 | 编译错误 | 走 fp16/bf16 |
| D6 | `__shfl_down_sync` 在 offset 32 处复制而非交换，warp-64 六轮归约返回 2× 正确值 | **是** | 用 5 轮 |
| D7 | Conv1d 在 >~1.62e9 元素输入上静默损坏首尾 `(B−48)` 个 batch | **是** | 批切片（已用于 harness） |

**D1、D3、D6、D7 是危险类**：程序报告成功却返回完全错误的结果。D6 尤其反转了标准指导（`offset = warpSize/2` 的推导在此 SDK 上是**错误**模式）——任何 MACA 升级后都应复测。

### 4.1 D7 的处理（唯一的平台失败）

L1 P76（膨胀/步进 1-D 卷积，输入 2,147,450,880 元素）：

- 全张量前向与 im2col+GEMM 仲裁器（从不调用 conv 内核）相差 maxdiff 2.0–2.7，99.98% 输出元素错误
- 同一卷积**逐 batch 执行则完全精确**（8.3e-7）；阈值在 1.61B–1.64B 输入元素之间，损坏 batch 呈对称首/尾块——**寻址/平铺特征，而非浮点非确定性**
- `torch.backends.cudnn.deterministic=True` **不能修复**（路径是确定性的但错误）
- **Harness 层缓解**：对 >1.62e9 元素的 `nn.Conv1d.forward` 施加批切片守卫（`shims/sitecustomize.py`，环境层实现，上游源码零改动），P76 在其下 5/5 通过。例试验 L1 P67、L3 P29（仅有的其它 Conv1d 题）均 5/5 通过，回归范围为精确零

---

## 5. 仓库结构

```
maca-harness/                 455 个 tracked 文件
├── KernelBench/              上游源码，SHA256 记录于 SOURCE_SHA256.txt，零改动
├── run_batch.py              批量正确性执行器
├── perf-baseline.jsonl       270 题基线计时（核心交付物）
├── align-final.jsonl         L1-3 官方 gate 结果
├── l4-results.jsonl          L4 结果 20/20
├── shims/                    环境层兼容：openai/litellm/pydra/modal stub
│   └── sitecustomize.py      eval 模式补丁 + D7 Conv1d 守卫
├── docs/conv-family/         conv 否定结果：5 个自包含探针 + 原始 JSON + write-up
├── optloop/                  Phase II 内核实验（wmma/mcblas/mctlass 探针）
├── probe_p76_{verdict,scope,reconcile}.py   D7 定位探针
└── rerun_l4_per_problem.py   L4 隔离测量驱动
```

**可复现性**：所有 conv 探针已实跑验证——重跑后 tracked JSON 位级不变，输出值与 committed 完全一致。`probe_conv_algo.py` 的 docstring 明确标注其参考为 TF32 修正前的第一代读数，避免读者重蹈 Finding 4 的量尺错误。

---

## 6. 未决事项

按 backlog 五项清单：

| # | 事项 | 状态 |
|---|---|---|
| 1 | 推送提交 | ✅ |
| 2 | 报告 MACA 工具链缺陷（D1–D7） | ✅ |
| 3 | 性能调优（58/82 + conv 线关闭） | ✅ |
| 4 | **Agent loop 批量 rollout → RL 训练** | ❌ **未开始** |
| 5 | 验证层注意事项（已记录） | ✅ |

唯一未决项是第 4 项（stage-1 batch rollout → RFT/PPO → KernelBench 评测），属于另一 session 作用域。性能线已达到当前 SDK 的可行上限——剩余 24 题 GEMM 的边界问题是每题的关键 matmul 恰是误差注入层，选择性 fp32 路由会花掉唯一的加速来源；conv 提速需要厂商侧改动。

---

## 7. 提交历史与身份

maca-harness 共 **20 个 commit**，全部统一为 `Dryoung95 <215174080+Dryoung95@users.noreply.github.com>`（含一次 `filter-branch` 历史改写：修正 1 个 QQ 邮箱 + 14 个缺 ID 的 noreply 形式；树哈希一致，内容零变更）。kda-on-maca 共 **22 个 commit**，身份同样全部标准。两库均与 GitHub 远端同步（分叉 `0 0`），工作区干净，tracked 文件无任何敏感信息。

---

**结论**：作为 KernelBench 向 MACA 平台的迁移交付物，本仓库完整——正确性对齐官方标准、270 题基线建立、平台缺陷有据可查、性能边界经测量确认。剩余的 RL 训练线是后续工作，且需在 64 GB 配置上展开。
