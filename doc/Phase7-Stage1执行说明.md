# Phase 7：Pretrain Stage 1 执行说明

## 当前状态

**最终状态（2026-08-29）：No-Go，因硬件条件阻塞，正式 Stage 1 与 Stage 2 均停止执行。**

本文后续命令保留为实验归档，不再作为待执行指令。项目选择交付方法级 Demo，不使用
LoRA/QLoRA 替代论文全参数训练，也不使用 soak 临时权重充当正式 checkpoint。

预算 subset 已完成并锁定：320 个 train 基础表、40 个 validation 基础表；Stage 1 为
960/120 条三任务样本，Stage 2 为 320/40 条 sequential 样本。四任务真实多模态序列最大
1,534 tokens，prompt 最大 512 tokens，无截断。

正式 Stage 1 原定在通过 32-step Stage 1/2 soak 后启动。soak 使用与正式训练相同的模型、
loss、优化器与硬件配置，临时权重不进入正式训练。

## 32-step soak

```bash
python -u -m reproduction.train.run_stage12_soak \
  --run-name phase7_stage12_soak_v1
```

配置：Stage 1 24 steps，保存并重载；Stage 2 8 steps，保存并重载；两个阶段均为
full-parameter、BF16、SDPA、batch size 1、gradient checkpointing、学习率 `4e-5`。
每阶段最长允许 1,800 秒，用户按 Ctrl+C 时父进程会终止当前 GPU 子进程。

最终报告：

```text
reproduction/results/training_feasibility/phase7_stage12_soak_v1.json
```

报告只有满足以下全部条件时才为 `decision=go`：32/32 steps、loss 有限、两个 checkpoint
保存/重载成功、清理后 allocated drift 不超过 0.25 GiB，且含 validation 与 20% 余量的
Stage 1 + Stage 2 预测总时间不超过 7.5 小时。

实际报告为 `decision=no-go`。Stage 1 完成 24/24 steps、checkpoint 保存和重载；Stage 2
完成 2/8 steps 后在第 3 步 backward 返回 `cudaErrorUnknown`。同期 WSL 内核日志出现
`dxgkio_make_resident: Ioctl failed: -12`，表明当前 8GB GPU + WSL 的显存/共享内存驻留
无法为正式长时间全参数训练提供足够可靠性。

## 正式 Stage 1

以下命令因最终 No-Go **禁止执行**，仅保留为历史配置：

```bash
python -u -m reproduction.train.stage_trainer \
  --stage stage1 \
  --run-name stage1_budgeted_e1_lr4e5
```

该命令从原始 HunyuanOCR v1.0 开始，训练全部 960 条 Stage 1 数据 1 epoch，并计算 120 条
validation loss，随后保存、重载和短生成。正式模型输出到：

```text
reproduction/checkpoints/stage1_budgeted_e1_lr4e5/
```

正式训练未启动，因此不生成正式 Stage 1 checkpoint，也不在固定 TableVerse-100 subset
上填报 Stage 1 指标。`reproduction/checkpoints/soak/phase7_stage12_soak_v1_stage1/` 只是
诊断 checkpoint，不作为模型成果。Phase 8 因缺少正式 Stage 1 checkpoint 且自身 soak
失败，同样标记为硬件阻塞。
