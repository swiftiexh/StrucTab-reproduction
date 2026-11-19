# Phase 0 运行环境记录

验证日期：2026-08-28。

```text
OS: WSL2 / Ubuntu
GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
Driver: 560.94
Driver-reported CUDA: 12.6
Python: 3.10.20
PyTorch: 2.13.0+cu126
TorchVision: 0.28.0+cu126
Transformers: 5.16.1
Accelerate: 1.14.0
```

已验证：

- `torch.cuda.is_available() == True`；
- 单卡可见；
- BF16 可用；
- CUDA tensor 运算成功；
- `HunYuanVLConfig`、`HunYuanVLProcessor` 和
  `HunYuanVLForConditionalGeneration` 可从本地 checkpoint 配置加载/导入；
- `pip check` 无 broken requirements。

按项目决定不生成完整环境 lock；后续新增训练、RL、渲染和 Demo 依赖时继续记录实际版本。

