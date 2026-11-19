# StrucTab 方法级小规模复现

本目录存放复现代码。`../StrucTab/` 保留为官方源码参考；除已记录的逻辑列数
bugfix 外，新增的数据转换、训练、评测和 Demo 代码均放在本目录。

## 文档索引

- [实现计划](../doc/实现计划.md)：阶段路线、状态与验收条件。
- [实验结果](../doc/实验结果.md)：跨阶段总表、分层指标和异常样本复核。
- [实现来源台账](../doc/实现来源台账.md)：每阶段复用、未复用和自行实现的内容。
- [Debug 记录](../doc/debug记录.md)：源码 bug、边界行为、修复与验证。

每个 Phase 收尾时同步更新 README、实验结果和实现来源台账；发现 bug 时另行更新
Debug 记录。

## 固定约束

- 全局随机种子：`42`。
- SynthTabNet、FinTabNet：仅用于预训练。
- PubTables-1M：仅用于 SFT / RL，两个 split 必须互斥。
- TableVerse-5K：只用于评测和评测样例展示，禁止进入任何训练阶段。
- 样本以稳定 `sample_id` 为主键，不能只使用 `image_path`。
- Stage 2 使用官方三段输出协议，协议常量集中在 `common/protocol.py`。
- 1D Probe 的已知额外后缀边界暂时保留官方行为。

## 路径规则

配置文件中的项目路径均相对于项目根目录解析，不写入个人绝对路径。公共路径入口在
`common/paths.py`。

## 测试

从项目根目录执行：

```bash
python -m unittest discover -s reproduction/tests -v
```

Phase 0 共 13 个测试，覆盖基础配置一致性、随机种子、项目相对路径、稳定
`sample_id`、数据隔离、三段输出协议和官方逻辑列数 bugfix。使用当前 ID 规则扫描
TableVerse-5K 的 4,955 条记录得到 4,955 个唯一 ID；重复 `image_path` 不再造成主键碰撞。
Phase 1 新增固定子集、推理和 TEDS 评测辅助测试。Phase 2 新增 Schema、HTML 恢复、
OTSL/合并网格对齐与断点续跑测试。Phase 3 新增统一结构解析器的正常与异常边界测试，
Phase 4 新增 Stage 1/Stage 2 构造及三段协议测试，当前共 67 个测试。
Phase 5 新增固定抽样、分层 split、pool 统计和完整性验收测试，Phase 5 完成时共 72 个测试。
Phase 6 新增训练 target masking、策略配置、精确探针选样、OOM 分类与外层超时守护测试，
Phase 7 新增预算 subset、通用 Stage trainer 与 soak 验收测试。Phase 9 新增 PubTables
质量规则、确定性分层选择和 SFT/RL 互斥划分测试。Phase 10 新增官方 reward 原语、
分量隔离、失败降级、确定性、renderer、Anchor cache 与审计用例测试，当前共 110 个测试。
Phase 13 新增 Demo 目录、代理 Reward 派生/解释、1D sequence、Anchor Destylization、
HTML 安全清洗、上传白名单和 API 契约测试，当前共 125 个测试。

## 项目交付边界

项目最终收束为 **StrucTab 方法级 Demo**。Phase 0～6 以及 Phase 7 的预算数据、训练器和
Go/No-Go 验收工具均已实现；论文对齐的正式 Stage 1 与 Stage 2 全参数训练不再在当前
RTX 4060 Laptop 8GB + WSL 设备上执行。

32-step soak 中，Stage 1 完成 24/24 steps、保存和重载；Stage 2 在第 3 个训练样本的
backward 阶段出现 `cudaErrorUnknown`，同期 WSL 内核报告
`dxgkio_make_resident: Ioctl failed: -12`。这说明短探针可以运行，但依赖超额显存/共享
内存驻留的长时间训练无法满足“Stage 1 + Stage 2 在 7～8 小时内一次稳定完成”的项目
约束。因此：

- Phase 7：**因硬件条件阻塞**，未生成正式 Stage 1 checkpoint；
- Phase 8：**因硬件条件阻塞**，且依赖未完成的正式 Stage 1 checkpoint；
- Phase 11：**Go/No-Go 判定为 No-Go**，缺少正式 Stage 2/SFT checkpoint，且 RL 协同负载
  高于已失败的 Stage 2 soak，因此不执行 mini RL 探针或 optimization；
- Phase 12：固定 baseline 评测与 reward 组件审计已完成，但跨 Stage 1/Stage 2/SFT/RL
  的统一模型评测和消融因不存在正式 checkpoint 而阻塞，不执行也不填写虚构结果；
- soak checkpoint 仅为诊断产物，不作为阶段模型或效果结果；
- 不使用 LoRA/QLoRA 冒充论文的全参数训练结果。

详细证据见 [`doc/Phase7-Stage1执行说明.md`](../doc/Phase7-Stage1执行说明.md) 和
[`doc/debug记录.md`](../doc/debug记录.md)。

## 阶段结果

所有阶段都在同一份 100 条固定 subset 上比较。完整总表、分层结果和固定 failure
cases 见 [`doc/实验结果.md`](../doc/实验结果.md)。

| 阶段 | TEDS | TEDS-S | Exact Match | 多 Table 启发式命中† | 平均耗时/s | 峰值显存/GiB |
|---|---:|---:|---:|---:|---:|---:|
| HunyuanOCR Baseline | 0.742921 | 0.824010 | 0.0300 | 2 | 34.5428 | 2.5225 |
| Stage 1（硬件阻塞，未正式训练） | — | — | — | — | — | — |
| Stage 2（硬件阻塞，未正式训练） | — | — | — | — | — | — |
| SFT（仅构造数据，不训练） | — | — | — | — | — | — |
| mini RL（No-Go，未执行） | — | — | — | — | — | — |

† 两个命中样本的原图本身包含多个表格，不能作为论文所述未闭合、无限循环式重复退化的
证据。它们仍属于固定 subset，但 Demo 已撤销 failure-case 标签和快捷入口。

## 单样本 baseline

```bash
python -m reproduction.inference.hunyuan_baseline \
  --image assets/datasets/TableVerse-5K/images/40046d1c73c40a0f22f039ca9490c93a.jpg \
  --sample-id tableverse-fb3c9300722ead28 \
  --max-new-tokens 512
```

结果默认写入 `reproduction/results/baseline/single_sample.json`，同时保留原始模型
输出、抽取出的完整 `<table>...</table>`、耗时、显存、token 数及截断/重复诊断。

## 固定 baseline subset

重新生成 seed 42 的 100 条固定评测清单：

```bash
python -m reproduction.data.build_baseline_subset
```

首次运行或从已有结果断点续跑使用同一条命令：

```bash
python -m reproduction.inference.run_baseline_subset
```

程序按 `sample_id` 跳过已经成功的样本，失败和中断中的样本会自动重试。可用
`--limit 5` 限制本次最多处理 5 条；只有确实要清空已有进度时才使用 `--restart`。
预测和运行指标分别保存到 `reproduction/results/baseline/predictions.jsonl` 与
`reproduction/results/baseline/metrics.json`。当前 `metrics.json` 已包含运行状态、
严格字符串匹配、TEDS/TEDS-S、主分层结果和多 Table 启发式记录。

当前 baseline 默认使用 `sdpa` attention，以避免高分辨率图片在 8GB GPU 上触发
vision eager attention OOM。若结果目录中仍是早期 eager 配置生成的进度，需要且只需
执行一次：

```bash
python -u -m reproduction.inference.run_baseline_subset --restart
```

后续中断恢复再执行不带 `--restart` 的普通命令。

## TEDS / TEDS-S

推理全部完成后，在一个终端启动官方算法的单 worker 服务：

```bash
python -u -m reproduction.eval.teds_server
```

服务只绑定 `127.0.0.1:18910`，不会启动官方脚本默认的 30 个端口。另一个终端可先
检查服务，再执行可断点续跑的正式评分：

```bash
python -m reproduction.eval.teds_client
python -u -m reproduction.eval.score_baseline
```

评分器会拒绝不满 100 条、存在失败项或混合 attention backend 的预测，因此不要在
baseline 推理仍在写入时提前运行。逐样本结果写入
`reproduction/results/baseline/teds_scores.jsonl`，最终均值写入现有 `metrics.json`
的 `evaluation` 字段。服务可在评分结束后按 `Ctrl+C` 关闭。

## 统一训练数据 Schema

Phase 2 的统一记录类型位于 `data/schema.py`，固定六个顶层字段：`sample_id`、
`source`、`split`、`image_path`、`html` 和 `metadata`。`image_path` 必须是项目相对路径，
`html` 必须是唯一、完整的外层 `<table>...</table>`，当前 `schema_version=1` 保存在
metadata 中。

三个数据源分别转换，默认目标是各 30 条 smoke 样本：

```bash
python -m reproduction.data.convert_synthtabnet
python -m reproduction.data.convert_fintabnet
python -m reproduction.data.convert_pubtables
python -m reproduction.data.validate_normalized
```

重复执行转换命令会读取已有 JSONL，并按 `metadata.source_key` 续跑；只有明确要删除对应
转换进度时才传 `--restart`。`--limit` 表示输出文件的目标总条数，设为 `0` 才转换全部
可用数据。转换产物、提取图片、`*_summary.json` 与 `*_failures.jsonl` 均在
`reproduction/data/normalized/`，属于可重建的本地数据产物，因此被 `.gitignore` 排除。

SynthTabNet 从 structure/cells 恢复 HTML，原图不重复拷贝；FinTabNet 直接使用
`html_table` 并从 Parquet 提取图片；PubTables 使用与 OTSL 对齐的 `html_restored`，
注入 cell tokens 后逐样本比较完整 merged-cell occupancy grid。当前真实 smoke 验收为
三源各 30 条、共 90 个唯一 ID、0 个转换失败，所有图片均可打开。

## 统一 HTML 结构解析器

Phase 3 的唯一入口是 `structure/parser.py` 中的 `parse_html_structure(html)`。它只建立
一次 occupancy grid，并从同一网格同时输出：

- `row_count` / `col_count`；
- 带 1-based 起止坐标、rowspan/colspan 的物理 `cells`；
- `merged_cells` 与官方 `<merge>(r1,c1),(r2,c2)</merge>` 字符串；
- `anchor_grid`、`is_valid` 和可追踪的 `errors`。

解析器不会返回重写后的 HTML，因此不会把 BeautifulSoup 的序列化结果替换原始训练
target。非法 span、span 重叠、不规则行宽、越界 rowspan、嵌套表格和缺失闭合标签均会
返回 `is_valid=False`，而不是静默修复。

对现有 90 条 normalized smoke 数据执行全量检查：90/90 有效，其中 66 条含 merge。
固定复杂样本审计可通过以下命令重建：

```bash
python -m reproduction.structure.audit_parser
```

审计从三个数据源各取 10 条合并表，共 30 条，结果全部满足 cell 矩形、grid anchor 和
merge 协议三者一致。机器可读清单位于
`reproduction/data/normalized/structure_parser_audit.json`。

## Stage 1 / Stage 2 数据构造

Phase 4 使用 `data/training_examples.py` 定义派生训练样本。Stage 1 对每个基础表生成三个
等量且独立的任务：

| `task_type` | prompt 目的 | target |
|---|---|---|
| `row_col` | 判断行列数 | `这是一个x行y列的表格。` |
| `merge` | 识别合并单元格 | 官方 merge 协议或无 merge 文案 |
| `html` | 表格解析 | 原始纯 `<table>...</table>` |

Stage 2 每个基础表只生成一条 `sequential` 样本，target 使用官方三个中文 section；不含
`<STRUCTURE>` / `<HTML>`。`common/protocol.py` 现在同时提供 formatter 和严格 parser，
训练数据、后续推理和 reward 对接应统一调用这里的协议常量与函数。

默认只读取允许用于 pretrain 的 SynthTabNet 与 FinTabNet smoke 数据：

```bash
python -m reproduction.data.build_stage_data
python -m reproduction.data.audit_stage_data
```

当前生成 60 个基础表对应的 180 条 Stage 1（每任务 60 条）和 60 条 Stage 2，共 240 个
唯一派生 ID，失败 0。全量 60 条 Stage 2 均被官方 Uni-TabRL reward parser 接受；seed
42 随机审计 30 组的三个 Stage 1 答案和 Stage 2 语义全部正确。输出位于
`reproduction/data/training/`，是可重建的本地数据产物，不提交版本库。PubTables 因用途
限制不进入本阶段预训练数据。

## Pretraining Pool

Phase 5 从 SynthTabNet 原始 `train` split 的 120,023 条中，以 seed 42 的稳定 SHA-256
优先级选择 10,500 个候选，其中 500 条作为长度/结构过滤备用；FinTabNet 使用本地两个
Parquet shard 的全部 6,867 条。执行入口为：

```bash
python -m reproduction.data.select_pretraining_sources
python -u -m reproduction.data.materialize_pretraining_sources --restart
python -u -m reproduction.data.build_pretraining_pool
python -u -m reproduction.data.validate_pretraining_pool
```

最终 pool 位于 `reproduction/data/pools/pretraining_pool.jsonl`，保持 Phase 2 的统一
Schema；轻量抽样/split 清单位于 `pretraining_pool_manifest.jsonl`。当前结果为：

- SynthTabNet 10,000，FinTabNet 6,865，共 16,865；
- train 16,022，validation 843，按 source 固定 seed 分层；
- complex 7,936、merged 6,183、simple 2,746；
- 16,865 个唯一 sample ID 与图片路径，全部图片解码成功；
- 2 条 FinTabNet 结构异常进入 rejected 清单，未混入 pool；
- 本地 HunyuanOCR tokenizer 统计的 Stage 2 target 最大长度为 5,811 tokens。

当前文本 target 硬上限为 8,192 tokens，构建过程从不截断 HTML。Phase 6 会把图像 token、
prompt 和显存纳入实测；如果 8GB GPU 要求更低的完整序列上限，必须显式生成过滤清单并
重建训练 pool，不能在 DataLoader 中静默截断。

## 训练可行性实验

Phase 6 使用 4 个基础表构造 12 条固定 Stage 1 探针，覆盖两个来源、三种任务以及
297～3,094 tokens 的完整多模态序列。最终固定训练策略为：HunyuanOCR BF16、SDPA、
batch size 1、全参数训练、gradient checkpointing；996,208,112 个参数全部可训练，
未使用冻结、LoRA 或 QLoRA。

最终 12-step 验收完成全部探针，loss 全部有限；完整模型保存为 1.9 GiB safetensors，
重新加载后对首条 row/column 样本生成“这是一个9行11列的表格。”。结果位于
`results/feasibility/checkpoint_12step_reload.json`，checkpoint 位于
`checkpoints/feasibility/checkpoint_12step_reload/`。

执行命令及各策略探针见 [`doc/Phase6-训练可行性执行说明.md`](../doc/Phase6-训练可行性执行说明.md)。
WSL 中 CUDA allocator 的 reserved/peak 数值可能超过 8GB 物理显存，并可能打印可恢复的
allocation warning；应以 step、loss、checkpoint 和 reload 是否真正完成判断实验状态。

## PubTables 高质量池与 SFT 数据

Phase 9 继续实现数据侧方法，但不执行实际 SFT。目标是完整转换本地 5,822 条 PubTables，
过滤缺失图片、非法结构、行列/标注冲突、低于 2×2、极端大表和超长 target，再以 seed 42
按 simple/merged/complex 分层稳定选择 4,000 条。最终固定划分为 SFT 3,200 条、RL 800
条，两者 base sample ID 必须完全互斥。

SFT 数据继续使用 Phase 4 的官方三段顺序协议，不含 `<STRUCTURE>` / `<HTML>` 包装。
完整执行命令与产物说明见
[`doc/Phase9-PubTables高质量池执行说明.md`](../doc/Phase9-PubTables高质量池执行说明.md)。
本阶段只生成数据，不加载训练模型，不生成 SFT checkpoint，也不填写 SFT 效果指标。

Phase 9 全量构建已完成：5,811 条成功转换与 11 条上游 OTSL 冲突合计覆盖全部 5,822 条；
质量筛选后 5,796 条合格，其中按 seed 42 分层选择 4,000 条。正式 pool 包含 complex
1,532、merged 865、simple 1,603；SFT/RL 为 3,200/800，ID 交集为 0。26 条明确拒绝由
11 条转换冲突、5 条少于 2 行、8 条超过 50 行、2 条超过 25 列组成。另有 1,796 条质量
合格但未被固定预算选中，它们不属于 rejected。

## Uni-TabRL Reward Pipeline

Phase 10 只实现并验证 reward pipeline，不执行 RL 训练。当前已直接复用官方 validity、
HTML 清洗、三段 response parser 和 1D Probe，按官方 `0.4/0.3/0.3` 权重聚合 validity、
structure、content，并自行补齐依赖隔离、本地 TEDS 接口、离线 HTML renderer、Anchor
HunyuanOCR adapter 与可续跑 reference cache。

本地 renderer smoke 已验证相同 HTML 首次渲染、第二次缓存命中，得到 258×222 RGB PNG；
Anchor smoke 已完成“HTML → PNG → HunyuanOCR → HTML”，输出表格完整、行列与 anchor grid
一致。`<th>` 被 OCR 规范化为 `<td>`，因此不满足字符串完全相等，但不影响本次结构核验。

已从 Phase 9 的 800 条 RL split 中以 seed 42 固定选择 6 条审计 reference，
simple/merged/complex 各 2 条，target 均不超过 1,024 tokens。选择清单位于
`data/reward/anchor_audit_selection.jsonl`，SHA-256 为
`fd5c9eb370e877653c1741d4859a18d3a6082d70a14200e4b10ecb7a11305907`。离线 Anchor cache
已完成 6/6、失败 0、耗时 52.9328s，SHA-256 为
`476203859f9747049f05de4867b8955ba7ca5741ace5cfc8fefd536448d7f164`。四类最终 reward
审计入口为 `reward/run_reward_audit.py`。四类审计均完成且没有 fallback：exact=1.000000、
content error=0.992683、structure error=0.479699、merge declaration error=0.600000。
结果位于 `results/reward/reward_audit.json`，SHA-256 为
`24cf13b07f1a1ebd89cddde158bcd23b50b924f1fb2768e9c00dad2ad5b08f51`；当前全项目 110 项
测试全部通过。

## 本地表格识别 Demo

Phase 13 实现单页 FastAPI Demo。页面只围绕表格识别：支持 100 条固定 TableVerse 示例、
结构/内容筛选、随机/前后切换、图片放大、实时 Baseline 推理、双向居中的上下预测/参考
表格、结构诊断及实时方法级代理 Reward。Reward 面板展开 Validity 比对、完整 1D Probe
sequence、首次错误位置和加权算式；Content 直接采用真实 Anchor-Guided Destylization。
Demo 不读取 Phase 1
缓存预测或指标；点击识别后才现场生成预测，并在本地运行官方 TEDS/TEDS-S 与 Reward。
100 条 GT 中有 99 条通过严格结构 parser、具备 Reward 计算前提；
`tableverse-8ca5eaf2d36eddbb` 的 GT 严格结构无效，因此实时结果中会明确显示 Reward 不可用。

Baseline 支持对固定样例和 JPEG/PNG/WebP 上传图片实时推理；模型延迟加载且 GPU 并发
固定为 1。上传图片没有 GT，不计算 Reward，也不展示“模型原始输出”。Stage 1、Stage 2、SFT 和 RL 按钮保留，
但均显示真实状态并禁用推理，不生成模拟结果。

一次 TableVerse 识别会顺序运行三次 OCR：原图 Baseline、预测 HTML 的 Anchor OCR、参考
HTML 的 Anchor OCR。Demo 显式采用 `should_de_stylize=True` 的方法展示假设；两份
Destylized HTML 之间的 TEDS 作为 Content 分量。原始 TEDS/TEDS-S 仍现场计算，但只作为
独立 TableVerse 评测指标，不参与代理 Reward。由于 Baseline 没有真实生成 Stage 2 的行列与
merge 分析，最终加权值明确标为方法级代理计算，不声称是已完成 RL 训练后的正式 Reward。

启动命令：

```bash
python -u -m reproduction.demo.app
```

浏览器访问 `http://127.0.0.1:8000`。完整验收步骤见
[`doc/Phase13-Demo执行说明.md`](../doc/Phase13-Demo执行说明.md)。
