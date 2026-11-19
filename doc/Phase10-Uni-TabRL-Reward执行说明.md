# Phase 10：Uni-TabRL Reward 执行说明

本阶段实现和验证 reward pipeline，不执行 RL optimization。

## 已完成

- 直接复用官方 validity、response parser、HTML 清洗和 1D Probe。
- 使用官方权重：Validity 0.4、Structure 0.3、Content 0.3。
- 本地 Chromium renderer 与 HunyuanOCR Anchor smoke 通过。
- 从 Phase 9 RL split 固定 6 条审计 reference，每个复杂度 2 条。
- 6 条 reference destylization cache 全部成功，失败 0。

## 最终 Reward 审计

终端 1 启动单进程官方 TEDS/TEDS-S 服务：

```bash
python -u -m reproduction.eval.teds_server
```

保持终端 1 运行，在终端 2 执行：

```bash
python -u -m reproduction.reward.run_reward_audit
```

审计固定使用第一条成功的 merged reference，依次构造：

1. `exact`：全部正确；
2. `content_error`：只替换首个非空单元格文本；
3. `structure_error`：删除最后一行并同步预测行数；
4. `merge_declaration_error`：HTML 正确，但前置分析错误地声明无 merge。

结构错误采用删除末行，不采用追加末行，因为官方 1D Probe 的已知边界不会惩罚预测末尾
额外结构。该边界按项目决定保持官方现状。

结果逐条原子写入：

```text
reproduction/results/reward/reward_audit.json
```

审计结束后，可在终端 1 按 `Ctrl+C` 停止 TEDS 服务。两条命令均不需要联网。

## 最终结果

审计使用 `pubtables-87379370e175fe5d`，4/4 用例成功，fallback 为 0：

| 用例 | Validity | Structure | Content | Total reward |
|---|---:|---:|---:|---:|
| exact | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| content error | 1.000000 | 1.000000 | 0.975610 | 0.992683 |
| structure error | 0.000000 | 0.794118 | 0.804878 | 0.479699 |
| merge declaration error | 0.000000 | 1.000000 | 1.000000 | 0.600000 |

结果文件 SHA-256：
`24cf13b07f1a1ebd89cddde158bcd23b50b924f1fb2768e9c00dad2ad5b08f51`。
