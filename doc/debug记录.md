# StrucTab Debug 记录

本文档持续记录复现过程中发现的官方源码问题、最小复现、处理决定、代码修改和验证结果。

## 2026-08-28：训练数据逻辑列数未考虑 span

### 状态

已修复，并增加回归测试。

### 位置

```text
StrucTab/code/training_data/get_sequencial_reasoning.py
parse_html_table()
```

原实现使用每行实际出现的 `td` / `th` 标签数作为列数：

```python
col_count = max((len(r) for r in parsed), default=0)
```

该值是物理 cell 数，不是考虑 `rowspan` / `colspan` 后的逻辑列数。例如：

```html
<table><tr><td colspan="3">A</td><td>B</td></tr></table>
```

原实现返回 2 列，正确结果应为 4 列。

### 修改

`parse_html_table()` 现在维护 occupancy grid：

- 跳过由上方 `rowspan` 占据的位置；
- 将每个 cell 的 `rowspan × colspan` 区域写入网格；
- 取网格最大宽度作为逻辑列数；
- 拒绝非正整数 span；
- 拒绝相互重叠的 span。

为保持调用方兼容，返回值中的 `parsed` 仍是原来的物理 cell 文本列表；本次只修正 `col_count` 的语义。

### 回归测试

新增：

```text
reproduction/tests/test_official_training_data.py
```

覆盖：

- 普通矩形表格；
- `colspan` 逻辑列数；
- `rowspan` occupancy 与 `colspan` 组合；
- 非正整数 span。

验证结果：4 个回归测试全部通过。另对当前两个 FinTabNet Parquet shard 的
6,867 条 HTML 全量执行解析检查，结果为：

```text
成功解析：6867
解析失败：0
新旧列数不同：2
```

两条被修正样本的列数分别由 `2 → 3` 和 `6 → 7`，说明该问题会在当前实际训练数据中触发。

## 2026-08-28：1D Probe 不惩罚预测末尾的多余结构

### 状态

已确认，暂不修改，保留官方实现。

### 位置

```text
StrucTab/code/Uni_TabRL/reward/structure/structure_reward.py
one_d_probe_reward()
```

当前比较循环只遍历预测与参考 cell 序列的公共前缀：

```python
for i in range(min(total_cells, len(cells_pred))):
```

如果预测完整匹配所有参考 cell，随后又生成额外 cell 或额外行，循环不会检查多余后缀，仍可能返回 1.0。

### 保留原因

论文没有明确规定参考序列结束后出现额外结构时应采用何种计分方式。使用最大序列长度、加入 EOS 结构节点或增加额外完整性惩罚都会改变 reward 定义，因此当前不做推断性修改。

后续若实验中观察到该边界造成 reward hacking、重复生成或明显的训练偏差，再基于实际证据确定修复语义，并在本文档继续记录。

## 2026-08-28：Baseline 高分辨率样本触发视觉注意力 OOM

### 状态

已定位并修复推理配置与 OOM 断点处理；GPU 实测结果记录在本节末尾。

### 触发位置

```text
reproduction/inference/hunyuan_baseline.py
HunyuanBaselineEngine

reproduction/inference/run_baseline_subset.py
main()
```

样本 `tableverse-b341c11aeea2c1ea`（图像尺寸 `1520×660`）在 HunyuanVL
vision tower 的 eager attention 中尝试额外分配约 3.39 GiB，8GB GPU 发生 OOM。
这不是输出 token 上限导致的；异常发生在图像 prefill 阶段。

脚本还存在一个关联问题：捕获 OOM 后立即调用 `torch.cuda.empty_cache()`，但 CUDA
上下文已经报告 accelerator error，该调用再次抛错，使失败状态无法落盘。

### 修改

- HunyuanVL 本地实现声明支持 SDPA，因此统一将 baseline 默认 attention backend 从
  `eager` 改为等价但显存更低的 `sdpa`；保留命令行显式切换能力。
- run config 和每条预测记录 attention backend。
- 断点续跑前比较关键 run config，禁止混合 eager 与 SDPA 结果。
- CUDA OOM 后不再调用任何 CUDA 清理接口；先原子保存失败记录和 metrics，再以退出码
  2 结束进程，下次运行自动重试失败样本。
- 由于已有 29 条结果使用 eager，SDPA baseline 必须使用 `--restart` 从头统一重跑。

### 验证

在同一 RTX 4060 Laptop GPU 上对原 OOM 样本进行 SDPA 定向验证：

```text
输入尺寸：1520×660
输入 tokens：1043
输出 tokens：945
生成耗时：27.4836 秒
峰值显存：2.0650 GiB
完整 table：是
疑似截断：否
疑似重复：否
```

该样本完整生成并越过原来的 vision prefill OOM 位置，修复验证通过。完整 baseline
仍需使用 SDPA 配置从头运行，以保证 100 条使用相同 attention backend。

## 2026-08-28：Phase 2 opening-tag 判断误将 thead 识别为 th

### 状态

已修复，并增加回归测试。该问题只出现在本项目首版转换器开发过程中，不是官方源码
或 SynthTabNet 标注错误；错误产物已由 `--restart` 清除并重新生成。

### 位置与现象

```text
reproduction/data/html_reconstruction.py
reconstruct_html()
```

首版代码用 `str.startswith("<th")` 判断 `<th>` opening tag，因此 `<thead>` 也会命中。
SynthTabNet structure 以 `<thead>` 开始时，转换器会提前注入第一个 cell 内容，之后所有
cell 对齐错一位，并最终报错：

```text
structure contains more cell tags than cell annotations
```

首轮 150,000 条只读扫描因此全部进入 failure 记录，没有写入任何 normalized record。

### 修改与验证

opening tag 现使用边界正则 `^<t[dh](?:\s|>|$)`，只接受 `<td` / `<th` 标签本身，
不会再匹配 `<thead>`。新增 `test_thead_is_not_mistaken_for_a_th_cell` 回归测试。清除本次
错误转换进度后，SynthTabNet 重新转换 30/30 成功、失败 0；三源联合验收 90/90 成功，
全项目 45 项测试通过。

## 2026-08-28：FinTabNet 全量候选中发现两个结构异常样本

### 状态

已定位并从 Phase 5 pretraining pool 显式排除。没有修改原始数据，也没有放宽统一 HTML
parser；完整原因保存在 `pretraining_pool_rejected.jsonl`。

### 样本与原因

| sample ID | source key | 原因 |
|---|---|---|
| `fintabnet-12640a1c7f29331a` | `train-00006-of-00018.parquet:2313` | cell 19 的 rowspan 延伸到第 5 行，但表格只有 4 行 |
| `fintabnet-690c9d70d4c91808` | `train-00015-of-00018.parquet:2106` | 17 行表格中多数行只有 6 个逻辑位置，而最大行宽为 7，属于不规则行宽 |

### 发现过程与处理

Phase 2 的每源 30 条 smoke 数据未覆盖这两个样本。Phase 5 首次扫描本地 FinTabNet 全部
6,867 条时，统一 parser 拒绝第一个样本，随后 pool builder 改为将所有结构非法候选写入
rejected 清单并继续扫描，而不是整体中止或静默修复。最终确认异常共 2 条，其余 6,865
条 FinTabNet 进入 pool；16,865 条最终 pool 全量验收通过。

## 2026-08-28：Phase 6 单步实验的显存增量统计恒为零

### 状态

已修复，不影响训练过程、loss 或原始显存采样，只影响结果 JSON 中的派生字段
`allocated_memory_growth_gib`。

### 位置与原因

```text
reproduction/train/hunyuan_feasibility.py
run()
```

原实现用最后一步 allocated memory 减去第一步 allocated memory。单步可行性实验只有一个
step，两者是同一个值，因此 `allocated_memory_growth_gib` 必然为 `0.0`。

### 修改与既有结果解释

现改为“最后一步 allocated memory 减去模型加载完成后的 allocated memory”。因此后续
实验会表达从模型加载到完成训练 step 的显存增量。

既有 `full_1step` 的原始采样未被修改：模型加载后为 1.8557 GiB，step 结束时为
7.6176 GiB，按新口径可推导增量为 5.7619 GiB。该次训练状态和 loss 均有效。

## 2026-08-28：Phase 6 最长全参数样本在 CUDA OOM 后未返回 Python

### 状态

已定位无 checkpointing 的阻塞边界并增加外层超时守护；同一最长样本随后使用全参数
gradient checkpointing 成功完成，loss 为 0.109959。

### 现象与原因

固定探针中 3,094-token 的 FinTabNet HTML 样本使用全参数 BF16 训练时，CUDA allocator
报告物理设备 free memory 为 0，并在尝试分配约 14 MiB 时 OOM。此后 CUDA 调用没有把
异常控制权返回 Python，因此训练脚本内部的 `except` 无法执行，结果 JSON 停留在
`status=running` 且没有完成的 step。

### 防护修改

新增 `reproduction/train/run_feasibility_guarded.py`。它在独立、无 CUDA 上下文的父进程
中启动训练；超过 wall-clock 时限后终止整个训练进程组，并把结果原子更新为 `timeout`。
用户主动中断守护器时也会回收子进程并记录 `interrupted`。这项修改不改变模型、数据或
训练配置，只避免 CUDA/WSL 阻塞导致实验无限等待。

## 2026-08-28：Phase 6 多步训练的可恢复 allocator OOM 与张量生命周期优化

### 状态

最终实验成功，不构成训练失败。已保留一项降低瞬时张量共存的防御性优化，但不能声称
该优化促成了本次成功，因为成功进程启动时尚未加载修改后的代码。

### 最终事实

全参数 + gradient checkpointing 连续训练期间，CUDA allocator 先后报告申请
752,877,568 bytes 和 1,495,269,376 bytes 时 free memory 为 0。用户在中途日志出现后过早
判断为卡住；进程实际上自行恢复，完成 12/12 steps、写出完整模型、重新加载并正确生成。
因此这两条属于当前 WSL/CUDA 环境中的可恢复 allocator warning，不能记为 OOM 失败。

该成功进程使用修复前代码，说明上一轮局部变量引用并未阻止本轮验收完成，也不能把
warning 的根因确定为局部变量生命周期。PyTorch 报告的 reserved/peak 指标最高超过物理
8GB，表明 WSL 环境存在虚拟/共享内存口径；后续只在同环境内比较这些 allocator 指标。

### 保留的防御性优化

- 每步统计完成后立即再次将梯度设为 `None`，并显式删除 `batch`、`outputs`、`loss`。
- 记录释放后的 allocator 指标 `memory_after_cleanup`。
- 在 forward 前原子记录 `active_step`；若 CUDA 再次阻塞，结果文件可直接定位正在执行的
  sample，而不只知道最后完成的 step。
- 不在每步调用 `torch.cuda.empty_cache()`，避免改变正常 allocator 策略或在错误上下文中
  再次触发 CUDA 异常。

Python 在下一次赋值右侧完成前确实会保留旧 `outputs` 引用，因此显式删除可避免长序列
full-vocabulary logits 跨 step 短暂共存。该修改只缩短中间张量生命周期，不改变 forward、
loss、backward 或 optimizer update 的数值语义；它作为后续正式训练的稳健性优化保留，
但不作为已经由对照实验确认的 bugfix 宣称。

## 2026-08-29：Phase 7 预算 subset 的 validation 交叉分层不足

### 状态

已修复配额并增加逐样本筛选缓存；没有放宽显存 token 上限，也没有截断任何 target。

### 现象与原因

首轮真实 processor 筛选在累计评估约 575 个候选后停止。`validation/fintabnet/complex`
原计划选择 4 条，但其 14 个文本预筛候选中只有 3 条同时满足 prompt 不超过 512 tokens、
四种任务完整序列均不超过 1,536 tokens。

### 修改

- FinTabNet validation 的内部配额由 complex/merged/simple=`4/6/6` 调整为 `3/7/6`。
- validation 总数仍为 40，SynthTabNet/FinTabNet 仍为 `24/16`；只在同一来源内将一条
  complex 配额转移给 merged，不通过放宽安全边界强行填充。
- 新增按 token 上限命名的 `screening_cache_*.jsonl`，每个真实 processor 测量完成后立即
  flush+fsync。相同配置重跑会复用合格与拒绝结果，避免分层末尾失败或中断后从零扫描。

首轮代码尚无缓存功能，因此已经完成的约 575 条测量无法恢复，需要重新执行一次；之后
同配置运行具备筛选级断点复用能力。

## 2026-08-29：Phase 7 soak 的 Stage 2 出现 WSL CUDA residency 错误

### 状态

最终 Go/No-Go 为 `no-go`。项目决定停止正式全参数训练并交付方法级 Demo，不再通过重启
WSL 反复尝试 Stage 2，也不更换 optimizer 或改用 LoRA/QLoRA 来改变论文对齐口径。
Phase 7 与 Phase 8 均标记为因硬件条件阻塞。

### 实测结果

- Stage 1：24/24 steps，loss 全部有限，清理后 allocated 从 5.6240 GiB 到
  5.6240 GiB，无增长；checkpoint 保存、重载和短生成成功。
- Stage 1 第 8 步出现一次可恢复 allocator warning，随后完成剩余 16 步。
- Stage 2：前 2 步完成，清理后 allocated 均为 5.6271 GiB；第 3 步 backward 返回
  `cudaErrorUnknown`，不是 loss NaN，也没有清理后显存增长。
- 同期 WSL kernel log 记录 `dxgkio_make_resident: Ioctl failed: -12`，说明 DXG 无法将
  GPU 分配驻留到当时可用的显存/共享内存。

### 最终处理

- 不启动正式 960 条 Stage 1 和 320 条 Stage 2 训练。
- Stage 1 soak checkpoint 只保留为训练、保存与重载链路的诊断证据，不作为正式模型。
- 不产生 Stage 1/Stage 2 效果指标，也不声明相对 baseline 的模型提升。
- 保留预算数据、训练器、Go/No-Go 工具与失败报告，作为方法级 Demo 和硬件边界证据。

同时修复异常结果记录：今后将 `active_step` 复制为 `failed_active_step` 后再结束，避免
结果 JSON 丢失正在失败的 sample ID。

## 2026-08-29：Phase 9 自定义输出目录无法写入 summary

### 状态

已修复并增加回归测试；不影响正式默认路径，也没有污染正式数据产物。

### 现象与原因

30 条 PubTables 端到端 smoke 使用 `/tmp` 隔离输出时，pool、split 和 SFT JSONL 已成功
写出，但 summary 构造调用 `project_relative()`，强制要求所有路径位于项目根目录，最终
抛出 `ValueError`。CLI 允许自定义输出路径，因此该假设过强。

### 修改与验证

新增 `_portable_path()`：项目内部路径仍保存为可迁移的项目相对路径，项目外自定义路径
保存为解析后的绝对路径。增加项目外路径回归测试后，30 条输入选择 20 条、划分 16/4 的
完整 smoke 成功；随后全项目 93 项测试全部通过。

## 2026-08-29：PubTables cell 小于号文本未转义导致伪结构错误

### 状态

已修复 converter 的 cell 文本转义；需要用 `--restart` 重建 Phase 9 的全量 normalized
文件。原始标注中另外 11 条真实 OTSL 合并冲突不修复，将进入 high-quality rejected。

### 现象与定位

首次全量转换扫描 5,822 条，只输出 5,807 条并记录 15 个 `ValueError`：

- 11 条为 `xcel upper and left anchors disagree`；其 OTSL 中二维 merge 的上方和左方指向
  不同 anchor，且若干 `html_restored` 同时含重复 `rowspan` 属性，属于上游结构冲突。
- 4 条被严格 parser 报告嵌套 `<td>`/不规则行宽。原始结构模板本身有效，真正原因是 cell
  文本含 `<HS/GED`、`<dl`、`(3</≥3)`、`</ = 4.5` 等比较符号；恢复器直接拼接文本，
  `HTMLParser` 将其误识别为标签并吞掉后续 `</td>`。

### 修改与预期

`_cell_text()` 现在保留已经存在的 `&amp;` 等实体，同时将原始 `&`、`<`、`>` 转义为合法
HTML 文本。增加比较符号与实体不重复转义测试。四条伪失败应恢复成功；重建后的预期为
5,811 条 normalized + 11 条 conversion failures = 5,822 条来源记录完整记账。Phase 9
builder 同时读取两份文件，将 conversion failure 纳入 rejected，而不要求坏标注强行生成
训练 target。

针对 15 条原失败记录逐条复测，4 条比较符号文本现已成功转换，另外 11 条仍稳定返回
OTSL anchor 冲突，符合预期；全项目测试更新为 94/94 通过。

## 2026-08-29：Phase 10 工具沙箱禁止 Chromium socket/sandbox 操作

### 状态

代码已显式选择完整 Chromium，而不是 Playwright 默认的 headless shell；仍需由用户在正常
WSL 终端完成实际 render smoke。该现象暂不判断为项目代码失败。

### 现象与处理

Playwright 1.60.0 导入成功，浏览器缓存为 Chromium 1223。工具执行环境首次默认启动
`chromium_headless_shell` 时在 `sandbox_host_linux.cc` 报 `Operation not permitted`；显式
切换到 `playwright.chromium.executable_path` 后，完整 Chrome 又在 crashpad socket 的
`setsockopt` 处被工具沙箱拒绝。这两处均发生在页面创建之前，符合当前命令执行沙箱不允许
浏览器进程系统调用的限制。

renderer 保留 `--no-sandbox`（由 Playwright headless 默认添加）、禁用 GPU，并增加启动
失败时完整释放 Playwright driver 的异常清理。下一步由用户在共享工作区的正常 WSL shell
运行同一 smoke；只有那里也失败，才按真实 WSL 浏览器兼容问题继续 debug。

用户在正常 WSL shell 中随后成功完成 renderer smoke：首次 `cache_hit=False`、重复调用
`cache_hit=True`，输出为 258×222 RGB PNG。人工查看确认中文表头、两行内容、边框与空
单元格均正确渲染。因此此前两次 Chromium fatal 确认为工具沙箱限制，不是项目或 WSL
浏览器故障。

用户随后使用 `--restart` 完成全量重建：5,811 条转换成功，11 条失败全部且仅为
`xcel upper and left anchors disagree`，两者合计覆盖原始 5,822 条；normalized JSONL
SHA-256 为 `b8228d1daad9b9456326710d698a1adbbb2a687c9ea4698c65ad5766b8b6e463`。

## 2026-08-29：多 Table 启发式将多表原图误报为重复退化

### 状态

不修改 Phase 1 已保存的原始预测与 `metrics.json`，但撤销两条样本的“重复退化”定性；
Demo 已删除对应标签、快捷入口、多表状态卡和重复 HTML 标记，文档统一改称“多 Table
启发式命中”。

### 现象与原因

`diagnose_generation()` 的 `suspected_repetition` 只要检测到多个 `<table>` 开始或结束标签
就置为真。`tableverse-862b2ed904b503df` 与 `tableverse-7f4164e07024e949` 因而被初步列为
重复生成案例，但人工查看原图后确认它们本身分别包含多个独立表格。

该规则不能区分：

- 对多表原图生成多个闭合表格；
- 论文描述的缺少 `</table>`、自回归内容持续循环并最终命中输出长度上限。

因此此前“重复=2”只是机器启发式计数，不是经过确认的论文式重复退化。真正的重复退化
至少应联合检查未闭合标签、循环片段和命中 token 上限，不能仅由 table 标签数量判定。

### 验证

固定 100 条 subset 及正式 Phase 1 指标文件保持不变，保证实验可追踪；Demo catalog 不再
输出 `repetition_case` 或 `is_repetition_case`。相关回归测试同时验证这两个字段均不存在。
