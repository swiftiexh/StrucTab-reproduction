# Phase 6：训练可行性实验执行说明

## 当前状态

Phase 6 已于 2026-08-29 完成。CPU 多模态预处理 12/12 通过；GPU 最终运行完成
12/12 steps、完整 checkpoint 保存、重新加载与短生成。

固定探针包含 SynthTabNet/FinTabNet 各两个基础表，覆盖约 600/1,600 text-target token
两档；每个基础表生成 row_col、merge、html 三任务，共 12 条。实际完整多模态序列为
297～3,094 tokens，所有 target 都有非空监督区域，未执行截断。

## 通用约束

- 确保没有其他 GPU 进程占用显存。
- 所有策略使用 batch size 1、BF16、SDPA、seed 42。
- OOM 是预期实验结果，脚本会写 JSON 后以退出码 2 结束。
- OOM 后不在同一进程调用 CUDA 清理；下一条命令会启动全新 Python 进程。
- 前三个策略只跑 1 step，不保存数 GB checkpoint。
- 只有最终可行策略执行 12 steps 和 checkpoint reload。

最终可行策略是全参数 BF16 + SDPA + batch size 1 + gradient checkpointing。冻结 vision、
LoRA、QLoRA 未执行，因为无需降级。

## 最终结果

| 项目 | 结果 |
|---|---|
| 固定探针 | 12 条，row/column、merge、HTML 各 4 |
| 完整序列范围 | 297～3,094 tokens，不截断 |
| 可训练参数 | 996,208,112 / 996,208,112 |
| 12-step 状态 | `ok`，loss 全部有限 |
| checkpoint | 1.9 GiB，保存和重载成功 |
| reload 输出 | `这是一个9行11列的表格。` |

## 依次执行

### 1. 论文忠实的全参数 BF16

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy full \
  --run-name full_1step \
  --max-samples 12 \
  --max-steps 1
```

### 2. 全参数 + gradient checkpointing

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy checkpoint \
  --run-name checkpoint_1step \
  --max-samples 12 \
  --max-steps 1
```

### 2a. 全参数最长序列压力探针

若全参数短探针成功，优先验证论文一致的全参数方案能否处理固定探针中最长的完整序列，
而不是直接降级到 LoRA：

```bash
python -u -m reproduction.train.run_feasibility_guarded \
  --timeout-seconds 300 \
  --strategy full \
  --run-name full_longest_1step \
  --sample-id stage1-7645244ec7a608e3 \
  --max-steps 1
```

该 FinTabNet HTML 样本的完整多模态训练序列为 3,094 tokens；脚本不会截断 target。
守护父进程会在训练子进程超过 300 秒时先发送 TERM，必要时发送 KILL，并将结果状态
写为 `timeout`，避免 WSL/CUDA 在打印 allocator OOM 后一直不把异常返回 Python。

### 2b. 全参数 checkpointing 的 12 条稳定性与重载验收

最长样本实测成功后，以全参数 + gradient checkpointing 作为论文对齐的可行策略。依次
训练固定探针中的全部 12 条记录，保存完整模型，并重新加载执行短生成：

```bash
python -u -m reproduction.train.run_feasibility_guarded \
  --timeout-seconds 600 \
  --strategy checkpoint \
  --run-name checkpoint_12step_reload \
  --max-samples 12 \
  --max-steps 12 \
  --checkpoint-reload
```

gradient checkpointing 只用重计算换显存，不冻结参数；模型全部 996,208,112 个参数仍参与
训练。LoRA/QLoRA 仅在本项无法稳定完成时作为降级方案。

### 3. 冻结 vision tower + gradient checkpointing

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy freeze_vision \
  --run-name freeze_vision_1step \
  --max-samples 12 \
  --max-steps 1
```

### 4. 安装 LoRA 依赖

仅在前三项完成后执行：

```bash
python -m pip install peft
python -m pip check
python -c "import peft; print(peft.__version__)"
```

### 5. LoRA 三任务短探针

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy lora \
  --run-name lora_3step \
  --max-samples 12 \
  --max-steps 3 \
  --learning-rate 1e-4
```

### 6. LoRA 全 12 条、保存与重载

仅当 `lora_3step.json` 的状态为 `ok` 时执行：

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy lora \
  --run-name lora_12step_reload \
  --max-samples 12 \
  --max-steps 12 \
  --learning-rate 1e-4 \
  --checkpoint-reload
```

### 7. QLoRA 仅作为必要降级

只有 LoRA 仍 OOM 时才安装并尝试：

```bash
python -m pip install bitsandbytes
python -m pip check
```

```bash
python -u -m reproduction.train.hunyuan_feasibility \
  --strategy qlora \
  --run-name qlora_12step_reload \
  --max-samples 12 \
  --max-steps 12 \
  --learning-rate 1e-4 \
  --checkpoint-reload
```

## 结果位置

每次运行独立保存到：

```text
reproduction/results/feasibility/<run-name>.json
```

成功保存的 checkpoint 位于：

```text
reproduction/checkpoints/feasibility/<run-name>/
```

请在每条命令结束后保留完整终端输出。若发生非 OOM 错误，先停止后续策略并提供对应
JSON 的 `status`、`error_type`、`error` 和 traceback；不要直接跳到 QLoRA。
