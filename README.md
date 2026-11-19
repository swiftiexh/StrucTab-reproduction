# StrucTab 方法级小规模复现

本项目基于 **HunyuanOCR-1.0**，对论文 *StrucTab: A Structured Optimization
Framework for Table Parsing* 的关键工程链路进行小规模、方法级复现。项目重点不是宣称
复现论文最终 SOTA，而是实现并验证数据准备、结构任务构造、Baseline 评测、训练可行性、
Uni-TabRL Reward 与交互 Demo。

## 完成内容

- 固定 100 条 TableVerse 子集，完成 HunyuanOCR Baseline 推理和 TEDS/TEDS-S 评测；
- 统一 SynthTabNet、FinTabNet、PubTables 与 TableVerse 的数据 Schema；
- 实现严格 HTML 表格解析器，统一处理 `rowspan`、`colspan`、重叠、越界和不规则网格；
- 构造 Stage 1 三个独立任务及 Stage 2 顺序输出数据；
- 构建 16,865 张预训练池和 4,000 条 PubTables SFT/RL 高质量池；
- 实现全参数 BF16 训练探针、checkpoint 保存/重载和长序列 soak 验收；
- 实现 Validity、1D Probe Structure、Anchor-Guided Content Reward；
- 实现支持 TableVerse 示例、图片上传、实时识别和 Reward 可视化的 Demo；
- 为数据、结构、训练、Reward 与 Demo 编写 125 个测试。

## 关键结果

| 项目 | 结果 |
|---|---:|
| Baseline subset | 100 条，100/100 推理成功 |
| TEDS | 0.742921 |
| TEDS-S | 0.824010 |
| 预训练池 | 16,865 张 |
| 本地预算子集 | 320 train + 40 validation 基础表 |
| PubTables 高质量池 | 4,000 条：SFT 3,200 / RL 800 |
| 全参数训练 | 短程探针通过；两阶段正式训练 No-Go |

本地设备为 RTX 4060 Laptop 8GB + WSL。Stage 1 soak 完成 24/24 steps 并成功保存、
重载；Stage 2 在第 3 个样本 backward 时发生 CUDA/WSL 显存驻留错误。因此本项目没有
把诊断 checkpoint 作为正式模型，也没有使用 LoRA/QLoRA 冒充论文的全参数结果。

机器可读摘要见 [`reports/experiment_summary.json`](reports/experiment_summary.json)，详细
过程见 [`doc/实验结果.md`](doc/实验结果.md) 与
[`doc/实现来源台账.md`](doc/实现来源台账.md)。

## 代码结构

```text
reproduction/
├── common/       # 路径、协议、ID 与复现约束
├── data/         # 转换、Schema、数据池和训练样本构造
├── structure/    # 严格 HTML 结构解析器
├── inference/    # HunyuanOCR Baseline
├── eval/         # TEDS/TEDS-S
├── train/        # 可行性探针与 Stage trainer
├── reward/       # Uni-TabRL Reward Pipeline
├── demo/         # FastAPI + 原生前端 Demo
└── tests/        # 单元及集成测试
```

## 外部依赖与数据

本仓库不重新分发以下内容：

- StrucTab 官方仓库；
- HunyuanOCR 权重；
- TableVerse、SynthTabNet、FinTabNet、PubTables 数据；
- 本地 checkpoint、派生 JSONL、渲染缓存和上传图片。

代码按项目根目录下存在 `StrucTab/`、`assets/models/` 和 `assets/datasets/` 的布局编写。
官方源码版本、克隆方式和本地补丁见
[`third_party/README.md`](third_party/README.md)。依赖包见 `requirements.txt`；本次实验
实际环境版本见 [`reproduction/environment/README.md`](reproduction/environment/README.md)。

## 快速检查

准备好官方仓库及数据资产后，从项目根目录执行：

```bash
python -m unittest discover -s reproduction/tests -v
```

启动 Demo：

```bash
python -m reproduction.demo.app
```

更完整的命令、阶段状态和运行边界见
[`reproduction/README.md`](reproduction/README.md)。

## 项目边界

当前真正具备正式模型结果的只有原始 HunyuanOCR-1.0 Baseline。Stage 1、Stage 2、SFT
和 RL 已完成数据或程序链路，但未完成论文规模训练，因此仓库不提供这些阶段的性能数字。

