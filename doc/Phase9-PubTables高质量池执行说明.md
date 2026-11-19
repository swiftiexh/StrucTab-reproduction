# Phase 9：PubTables 高质量池与 SFT 数据执行说明

## 范围

本阶段继续构造高质量数据，但不执行实际 SFT。不会加载 HunyuanOCR 模型权重，不需要
GPU，不生成 checkpoint，也不在 TableVerse-100 上填写 SFT 指标。SFT 数据仍使用官方
三段顺序 target，以便完整展示 StructTab 的数据方法。

## 第一步：完整转换 5,822 条 PubTables

该命令可能耗时较长，由用户手动执行：

```bash
python -u -m reproduction.data.convert_pubtables \
  --output reproduction/data/normalized/pubtables_all.jsonl \
  --image-dir reproduction/data/normalized/images/pubtables \
  --limit 0 \
  --restart
```

转换支持按 `metadata.source_key` 断点续跑。首次全量运行发现 cell 文本转义 bug 后，代码
已经修复，因此当前这一次必须在上述命令末尾添加 `--restart`，清除旧的 5,807 条输出和
15 条旧失败记录后完整重建。若这次重建中途被用户中断，之后恢复时执行不带 `--restart`
的普通命令。既有图片会先验证再复用，不会重复覆盖。

全量审计已确认原始 shard 中有 11 条 OTSL/HTML 合并结构互相矛盾，必须明确拒绝。因此
修复 cell 文本转义后，完成结果应为：

```text
records: 5811
failures_this_run: 11
```

11 条 failure 应全部为 `xcel upper and left anchors disagree`。构建器会把 5,811 条成功记录
和 11 条转换拒绝合计核对为完整的 5,822 条来源记录，并把这 11 条写入正式 rejected
清单。若数量或错误类型不同，不要继续第二步，先把终端输出发给我分析。

## 第二步：构建高质量池和 SFT/RL 数据

```bash
python -u -m reproduction.data.build_pubtables_high_quality_pool
```

构建器会联合读取 normalized JSONL 和 conversion failures，拒绝未覆盖完整 5,822 条来源
记录的输入，并逐条执行：图片解码、统一 HTML parser、原标注行列一致性、
cell annotation 数量、最小 2×2、极端尺寸、三段 target 语义以及 8,192-token 上限检查。
所有 target 均完整保留，绝不截断。

默认使用 seed 42 按 simple/merged/complex 的合格分布比例稳定选择 4,000 条，再在每种
复杂度内分层划分 SFT 3,200 / RL 800。相同输入与配置重复构建会得到相同 ID 和 hash。

## 正式产物

```text
reproduction/data/high_quality/
├── pubtables_high_quality_pool.jsonl
├── pubtables_high_quality_rejected.jsonl
└── pubtables_high_quality_summary.json

reproduction/data/splits/
├── sft.jsonl
└── rl.jsonl

reproduction/data/training/
└── sft.jsonl
```

- `pubtables_high_quality_pool.jsonl`：4,000 条完整统一 Schema 基础样本。
- `pubtables_high_quality_rejected.jsonl`：未通过质量条件的样本及明确原因；未被 seed 选中
  但质量合格的样本不属于 rejected。
- `pubtables_high_quality_summary.json`：数量、分层、分布、质量阈值、路径和 SHA-256。
- `splits/sft.jsonl`：3,200 条轻量 SFT base manifest。
- `splits/rl.jsonl`：800 条轻量 RL base manifest，供后续 reward pipeline 使用。
- `training/sft.jsonl`：3,200 条可直接训练的三段 `TrainingExample`；本项目只构造、不训练。

## 验收条件

- 输入恰好 5,822 个唯一样本；
- pool 恰好 4,000 条；SFT/RL 恰好 3,200/800；
- `sft_ids.isdisjoint(rl_ids)` 为真；
- SFT examples 精确覆盖 SFT base IDs；
- 所有选中图片、HTML、三段 target 和结构语义有效；
- `training_status` 为 `not_run_by_project_decision`。

## 最终结果（2026-08-29）

全部验收条件已满足，Phase 9 数据侧工作完成：

- 来源记录 5,822 = normalized 5,811 + conversion failures 11；
- 质量合格 5,796，明确拒绝 26；
- pool 4,000 = SFT 3,200 + RL 800，ID 交集为 0；
- complex/merged/simple = 1,532/865/1,603；
- 3,200 条 SFT examples 精确覆盖 SFT split，旧包装标记为 0；
- Stage 2 target 最大 4,255 tokens，无截断；
- 实际 SFT 状态为 `not_run_by_project_decision`。

正式 hash 以 `reproduction/data/high_quality/pubtables_high_quality_summary.json` 为机器可读
依据，并已由独立脚本重新计算确认一致。
