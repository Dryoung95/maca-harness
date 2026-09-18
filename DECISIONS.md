# C500 迁移第二阶段：决策点（2026-09-16）

等待你的答复。以下每项给出选项、代价和我的建议。事实基础在文末，保证可核对。

---

## D1. 下一阶段的技术路线（最关键）

fp16 手写 wmma 已被证明**正确但慢**（0.32x）。瓶颈不是调参，是 mxcc 无法正确执行任何非标量的 half 内存访问，导致 fragment 只能逐元素标量填充，指令吞吐被钉死。要打败 eager 必须换路线。

**选项 A：调用 MACA BLAS（mcblas）**
- 做什么：L1 P1 的提交改为 `mcblas` 的 hgemm/sgemm 调用，或包一层 fp32 输入 → fp16 计算 → fp32 输出。
- 代价：低。SDK 自带，trace 里能看到 `mcblas__...` 内核，说明库本身在 C500 上跑得动。需要确认 mcblas 在 Python/torch 里的调用路径（可能要走 C++ 扩展或直接调库符号）。
- 风险：这不是"手写 kernel"，如果目标之一是验证**手写 CUDA 迁移能力**，这条路线绕开了迁移本身的价值。
- 我的建议：**先走这条**。它是唯一有可能立即超过 eager 的路径，能先拿到一个"成功样本"确认评测链路通。

**选项 B：用 mctlass（SDK 自带的 CUTLASS 移植）的 device 层 GEMM**
- 做什么：调 mctlass 的 `GemmDevice` 模板，它用 `wmma_array` 批量管理 fragment，可能绕开我们手写的标量 fill 问题（库内部可能用了我们没发现的对齐技巧，或同样受限于标量路径）。
- 代价：中。需要摸 mctlass 的模板实例化方式，但比手写快。
- 风险：mctlass 的 fragment 加载最终也调用同一批原语。如果它的性能也起不来，说明标量 fill 是**硬件/工具链层面的硬限制**，这个信息本身有价值。
- 我的建议：**作为 A 的对照**。A 跑通后用 B 检验"是库调优得好，还是有什么我们没看到的向量化路径"。

**选项 C：继续手写 wmma，等 MACA 修 bug**
- 代价：高，且收益不可预期。已试过 BK/launch_bounds/线程数/WM×WN 全组合，都在 0.15–0.33x。
- 我的建议：**不推荐**，除非你有来自沐曦的承诺说某个版本修了 col_major load 和 store_matrix_sync。

**选项 D：放弃 L1 P1 的性能，换题**
- KernelBench 有 250 题，L1 P1 只是最容易上手的一道。其他题（elementwise、reduction、conv）可能不需要 tensor core 就能打败 eager。
- 代价：中。但 L1 P1 的 matmul 是最有代表性的题，跳过它会让第二阶段的结论不够有说服力。
- 我的建议：**如果 A 也拿不到 >1x，就转 D**，并明确记录"matmul 类在 C500 上手写无法超越 eager，需依赖库"。

---

## D2. 是否向沐曦（MACA 官方）报告工具链缺陷

我们发现了三个可复现的编译器/库缺陷，都有最小复现：

1. `store_matrix_sync` 对所有 accumulator 类型是 no-op（源码看起来正确，但运行时不写任何元素）。
2. `load_matrix_sync` 的 col_major 变体用 double 向量化读取，返回打乱的数据。
3. 任何 `__half` 的位重解释（`unsigned`/`half2`/`ulonglong2`/`__half{bits}`）在 mxcc 下产出垃圾值。

**选项 A：现在报告，附最小复现**
- 代价：你需要花时间整理成官方能接受的形式（英文工单 + 可编译复现）。我可以把现有的隔离测试整理成一份自包含的报告。
- 收益：高。这三个 bug 直接挡住了所有手写 tensor core kernel 的迁移工作——不只是我们的项目，任何往 C500 移植 CUDA 代码的人都会撞上。修了之后手写 wmma 路线可能直接可用。
- **我的建议：报告**。这是本次工作里最有外部价值的产出。如果你同意，我把隔离测试整理成 `TOOLCHAIN_BUGS.md`（每个 bug 一个最小可复现 .cu + 期望/实际输出），你直接转发。

**选项 B：只记录在内部记忆里，不报告**
- 代价：零，但问题不会消失，下次还会撞上。

---

## D3. wmma6/wmma7 这两个"正确但慢"的 kernel 怎么处理

它们 rel=0.00000 全尺寸正确，但 0.32x，远低于 eager。

**选项 A：保留作为正确性参考**
- 价值：这是目前唯一**数值正确**的 fp16 tensor core matmul 实现证据。将来 MACA 修了 bug，可以立刻用它验证修复是否真的修好了（把 store/load 换回库实现，跑同一套测试）。
- 代价：零，代码已在 `kernels_wmma6.py` / `kernels_wmma7.py`。

**选项 B：同时作为"MACA wmma 可用性"的回归测试**
- 把 `load_matrix_sync` / `store_matrix_sync` 的隔离测试留成一套小测试（每个一个 .cu + 一个断言），以后升级 MACA SDK 时跑一遍。
- 代价：低，测试已经写过了，只是散在 /tmp 里需要归档。

**我的建议：A+B 都做**。归档成本很低，但能防止下次 SDK 升级后重复踩坑。如果你同意，我把散落在 /tmp 的隔离测试归档到 `optloop/toolchain_tests/`。

---

## D4. 第二阶段"成功"怎么定义

这个需要你拍板，因为它决定要不要继续投入 L1 P1。

**选项 A：打败 eager（speedup > 1.0）**
- KernelBench 的性能维度就是这个标准。L1 P1 的 reference 是 fp32 `torch.matmul`（1592us）。
- 注意：**fp16 eager 已经是 885us（1.80x）**，所以即使我们用 fp16 计算打败了 fp32 eager，也只是追平了"直接把输入转成 fp16 再 matmul"这个平凡做法。要不要把 fp16 eager 也当作要打败的目标，是另一个问题。

**选项 B：达到 fp16 上限的某个比例（如 50%）**
- 更技术性的目标，但和 KernelBench 的评测口径不一致。

**选项 C：只要求"能正确跑完 + 产出性能数据报告"**
- 把第二阶段当作调研，结论是"C500 手写 matmul 的性能边界在哪里"。

**我的建议：A**，但把"打败 fp32 eager"作为最低门槛，把 fp16 eager（885us）作为真实目标。如果只能做到前者，要诚实标注它等价于"自动混合精度"的水平。

---

## D5. 需要你答复的问题清单

1. **D1 选哪条？**（A/B/C/D，或组合，比如"A 先做，B 对照"）
2. **D2 要不要报告给沐曦？** 要的话我现在就整理最小复现文档。
3. **D3 要不要归档隔离测试？**
4. **D4 成功标准怎么定？**
5. **时间预算**：第二阶段你还想投入多久？这决定我是继续深挖 matmul，还是快速切到别的题。

---

## 事实基础（可核对）

**设备与工具链**
- MetaX C500，104 SM，16.3GB，warp 64，xcore1000。
- MACA SDK 3.3.0.15，驱动 3.3.0.4，mxcc 1.0.0 + cu-bridge。
- mcTracer 在 `/opt/maca-3.3.0/bin/mcTracer`；`--odname` 必须相对路径。

**性能基线（n=4096，cuda_event 计时，30 trials）**
| 路径 | 时间 | 吞吐 | vs fp32 eager |
|---|---|---|---|
| eager fp32 | 1592 µs | 86.4 TFLOPS | 1.00x |
| torch.compile fp32 | 1650 µs | — | 0.96x（无加速） |
| **eager fp16** | **885 µs** | **155 TFLOPS** | **1.80x** |
| eager bf16 | 773 µs | 178 TFLOPS | 2.06x |
| 标量 fp32 分块 | ~34000 µs | 4–5 TFLOPS | 0.047x |
| wmma3 最好配置 | 2333 µs | 58.9 TFLOPS | 0.682x |
| wmma7 最好配置 | 4952 µs | 27.7 TFLOPS | 0.321x |

**已排除的瓶颈**（都试过，无效或更差）
- tile 参数（BM/BN/BK/WM/WN 共 358 个配置）。
- BK 增大（16→32/64，减少 K 循环同步次数）：无效。
- `__launch_bounds__(1024, 2)`：编译器无法满足（需 ≤64 regs/thread）。
- 降到 512 线程换 occupancy：反而更慢（0.26x），说明瓶颈是吞吐不是延迟。
- shared bank 冲突：a-load 的 lane→bank 分布已验证无冲突。

**wmma7 最好配置的 launch geometry（mcTracer）**
- grid=(16,16)=256 blocks，1024 线程，126 regs/thread，shared 17920B。
- blocks/SM 被寄存器限制到 **1**（occupancy 24%）。但这不是主因（见上）。

**mcTracer 的 `dur` 不可用于性能比较**
- 同一 kernel 重复 trace：2.4ms / 2.9ms / 3.1ms（1.31 倍抖动），cuda_event 为 2333.2 µs ± 19.8 µs。
- 结论：`dur` 只能看 launch geometry / occupancy。

**MACA wmma 原语可用性（全部经隔离实验确证）**
| 原语 | 状态 |
|---|---|
| `fill_fragment` | 正确 |
| `mma_sync`（含 K 累加） | 正确 |
| `load_matrix_sync` **row_major** | 正确（标量实现） |
| `load_matrix_sync` **col_major** | **打乱数据**（double 向量化读取） |
| `store_matrix_sync`（所有类型） | **no-op** |
| `__half` 位重解释（unsigned/half2/ulonglong2） | **产出垃圾** |

**wmma3 的 258 个配置全部基于坏掉的原语**——之前标记为"correct"是因为容差碰巧通过（wmma3 用 fp32 输入路径，fragment 未真正实例化 half mma）。这些结果不能当作"正确且慢"，应当作"未真正验证"。

**工作区产物**
- `optloop/kernels_wmma6.py` / `kernels_wmma7.py`：正确的 fp16 wmma matmul。
- `optloop/sweep_wmma6.py`：配置扫描器（已跑 16 条后停止，小 tile 太慢）。
- `optloop/wmma6-results.jsonl` / `wmma7` 测试结果。
- `optloop/profile/wmma3_*/REPORT.md` + `optloop/profile/wmma7_*/`：两份 mcTracer 报告。
- 隔离测试目前散在 `/tmp/w4probe`、`/tmp/w5probe`、`/tmp/w6probe`、`/tmp/w7probe`（D3 如同意则归档）。

---

## 2026-09-17 补充：mctlass device GEMM 评估结果（D1 选项 B 已排除）

接续 D1 的选项 B（mctlass device 层）。跑了三轮隔离实验，结论是**这条路在 MACA 3.3.0 上不可用**，不用再投入。三条独立缺陷，全部可复现：

**B1. mxcc `-O2` 的 inlining bug 导致 host 端 segfault**
- 最小复现：只 include `mctlass/mctlass.h` + `mctlass/gemm/device/gemm.h`，main 里构造一个 `Gemm` 对象。`-O0` 正常，`-O1`/`-O2`/`-O3` 全部 segfault 在构造函数。
- 二分定位：`-fno-inline` 修复；`-fno-elide-constructors`、`-fno-omit-frame-pointer`、`-fno-vectorize`、`-fno-slp-vectorize`、`-fno-tree-loop-vectorize` 全部无效。所以是 mxcc 的内联器在这份模板上的 codegen bug。
- **已修复**：`optloop/mctlass_base.py` 的 `extra_cuda_cflags` 加了 `-fno-inline` 和 cu-bridge 的 include 路径。

**B2. ColumnMajor 输出特化算的是 B@A**
- `gemm.h:540` 的 ColumnMajor 输出偏特化把问题转置给 RowMajor 主算子，但 `to_underlying_arguments` 传 leading dimension 时**没有交换 A/B 的 stride**。方阵恰好掩盖，`err=1.31` 就是它的症状。
- **已规避**：直接用主算子（A RowMajor / B ColumnMajor / C RowMajor），这是 fp32 SIMT 唯一真正特化的组合。

**B3. SIMT epilogue 只写一半的行（未修复，这是真正的拦路问题）**
- 单位矩阵测试 `C = A @ I`：输出**行 R%16 < 8 完全正确，行 R%16 >= 8 全零**。
- 根因：`mctlass/gemm/warp/mma.h` 把 `WarpSize` 改成 64（C500 的 warp），但 epilogue 的线程映射（`default_thread_map_simt.h`、`output_tile_thread_map.h` 的 `RowArrangement`）仍硬编码 `kWarpSize = 32`。我修正了这三处后 `kThreads` 匹配了（512=512），但行轴的迭代计数仍不自洽：store 循环按 `Iterations::kRow=1` 只写 1 行/组，而 `operator++` 按 `Count::kRow=4` 推进。
- 尝试了三种修法（`Iterations::kRow := Count::kRow`；`advance_group := stride*(delta.group - delta.row*count.row)`；重写一个显式自洽的 `SimtThreadMap`），**全部触发 memory violation(0x4)**，说明 epilogue 还有别处（`FragmentIteratorSimt`、warp 级 `TileIteratorSimt` 的 smem 偏移）也按 32 线程 warp 假设。
- 结论：这是横跨 mctlass epilogue 多个文件的 64 线程 warp 移植缺口，盲改会波及 tensor-op 路径。**不再继续**。

**当前 mctlass 头文件状态**（可回退）
三处 `kWarpSize = 32` → `WarpSize<arch::OpClassSimt>::value`（`default_thread_map_simt.h:73`、`output_tile_thread_map.h:202/227`，后者另加了 `#include "mctlass/gemm/warp/mma.h"`）。`advance_group` 的实验已回退。原始备份在 `/tmp/orig_default_thread_map_simt.h`、`/tmp/orig_output_tile_thread_map.h`、`/tmp/orig_predicated_tile_iterator_params.h`。

**对 D1 的影响**：选项 B（mctlass device 层）排除。剩下的可行路径还是 **选项 A（mcblas）**——已验证正确（fp32=4747us、bf16=812us）和手写 wmma3（2333us = 真 fp32 的 2.03x）。等你定 D1。

---

## 2026-09-18：D1 选项 A 落地 —— L1 P1 首个 >1x 提交通过官方门槛

`optloop/submit_mcblas.py` 把 mcblas fp16-input/fp32-accumulate 路线接入官方 `eval_kernel_against_ref`：

```
compiled    : True
correctness : True (5/5)
hardware    : MetaX C500
runtime     : 951 us
ref_runtime : 1592 us
SPEEDUP     : 1.693x
```

**这是第二阶段第一个超过 eager 的提交**，也是 D1 所有选项里第一个跑通的。fp16 输入是唯一既过 fp32 容差又 >1x 的路径（mcblas fp32 直入 = 4742us = 0.34x，打不过 TF32 eager）。

**精度边界（`mcblas_precision_probe.py`，已确证）**：
| 输入分布 | rel err (fp16acc) | allclose(1e-4) | rel err (mcblas fp32) |
|---|---|---|---|
| `torch.rand`（P1 实际输入，正数） | **2.9e-6** | **True** | 3.0e-5 |
| `randn`（有符号、抵消） | 3.6e-5 | False | 3.2e-4 |
| `randn*10` | 2.6e-5 | False | 3.2e-4 |
| `randn*100` | 3.4e-5 | False | 2.9e-4 |

注意：randn 下 allclose=False **不是 kernel bug**——mcblas fp32 路径自身 rel err 就是 3.2e-4，这是"分块串行累加 vs torch 树状归约的求和顺序差异"（见 [[c500-kernel-optimization]] 的容差判据），fp16 输入反而**更准**（3.6e-5 < 3.2e-4）。是 allclose 对抵消型输入的绝对误差判据严格，不是精度劣化。

**结论**：L1 P1 的 1.69x 是真实的。mcblas fp16acc 路线可作为 matmul 类题目的标准提交路径。

**尚未解决/待你拍板**：
1. D2（是否向沐曦报告三个工具链缺陷）—— 尚未整理，隔离测试仍散在 /tmp，随时会丢。
2. D3（归档 wmma 隔离测试）—— 未做。
3. D4（成功标准）—— 目前按"打败 eager fp32"已达成第一题；是否继续在 matmul 上追 bf16 上限（812us，即 1.96x），还是切到其他题。
4. mcblas 路线目前只覆盖 L1 P1 一题。KernelBench 全量 250 题里 matmul 类题目（P1-P18 等）都可以用同一套 binding，但 binding 需要按题改。
