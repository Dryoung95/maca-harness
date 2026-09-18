# KernelBench → 沐曦 C500 迁移:第一阶段状态(2026-09-15)

## 源码
- KernelBench main 分支 tarball(sha256 见 SOURCE_SHA256.txt),落盘于 `/data/cuda-harness-migration/KernelBench`
- 上游:https://github.com/ScalingIntelligence/KernelBench (ICML'25)

## 环境(全部验证通过)
| 组件 | 版本/路径 | 状态 |
|---|---|---|
| GPU | MetaX C500, 104 SM, 16.3GB, mp_100 (xcore1000) | ✅ |
| MACA SDK | 3.3.0.15 (/opt/maca-3.3.0) | ✅ |
| 驱动 | 3.3.0.4 (/dev/mxcd) | ✅ |
| 编译器 | mxcc 1.0.0 + cu-bridge(cucc CUDA 语法桥) | ✅ |
| PyTorch | 2.8.0+metax3.3.0.2 (CUDA API 映射到 MACA) | ✅ |
| 设备访问 | coder 已加入 video 组(此前 GPU init 失败的根因) | ✅ |

## 已打通的链路
1. **构建闭环**:`torch.utils.cpp_extension.load_inline` 走 cu-bridge → cucc → mxcc,自动加 `-DUSE_MACA --offload-arch` flags,编译 CUDA 语法源码为 C500 fatbin 并加载
2. **正确性闭环**:`kernelbench.eval_kernel_against_ref` 全流程——随机输入、ref/new 双跑、allclose 比对、多 trial
3. **官方 CLI**:`scripts/run_and_check.py`(ref_origin=kernelbench, eval_mode=local)
4. **计时**:cuda_event timing 可用(torch.compile 亦可用)
5. **已验证 kernel 特性**:shared memory、__syncthreads、dim3 grid、<<<>>> launch 语法

## 验证记录
- 官方 example(elementwise_add):compiled=True correctness=True (3/3)
- Level1 P1 方阵乘法(手写 tiled shared-mem kernel):correctness 3/3, speedup 0.05x vs eager(预期,naive kernel)
- Level1 P10 3D 张量乘法:correctness 5/5(CLI 全流程,含 ref 计时)
- Level1 P1/2/3/14 ModelNew sanity:全过

## 已知限制(非迁移缺陷)
- **显存**:C500 16.3GB < KernelBench 部分题目的输入规模(如 P37 输入 7.5GB,P19 ReLU 输入 6.4GB;这些题在 16GB L4/T4 上同样跑不了,需要 48GB L40S/H100)。分类:invalid_infrastructure(硬件容量),非 backend_error
- gpu_arch 配置项沿用 NVIDIA 架构名(Ada 等),经 cu-bridge 转译为 xcore1000,无需改动即可工作
- performance(加速比调优)属于第二阶段,本次不评估

## 用法
```bash
source /data/cuda-harness-migration/env.sh
cd /data/cuda-harness-migration/KernelBench
python3 scripts/run_and_check.py ref_origin=kernelbench dataset_src=local \
  level=1 problem_id=10 kernel_src_path=<你的kernel.py> eval_mode=local
```
