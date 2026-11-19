# Phase 13：本地表格识别 Demo 执行说明

## 启动

在项目根目录和 `structab-repro` 环境执行：

```bash
python -u -m reproduction.demo.app
```

看到 Uvicorn 启动信息后，在浏览器访问：

```text
http://127.0.0.1:8000
```

服务默认只监听本机，不需要联网。结束时在终端按 `Ctrl+C`。

## 页面验收

1. 默认选中 HunyuanOCR Baseline 和一条 TableVerse 样例，右侧显示“等待实时识别”；选择样例不会读取 Phase 1 缓存，也不会加载 GPU。
2. 分别尝试结构与内容筛选；随机按钮只从当前候选中选取且不连续重复。
3. 点击左侧缩略图，确认可弹出完整图片；按关闭按钮、背景或 `Esc` 均可退出。
4. 点击“识别”，确认现场依次完成原图 Baseline、预测 Anchor OCR、参考 Anchor OCR；正式 Phase 1 文件不会被覆盖。
5. 确认预测表格与参考表格按上下布局展示；小表在水平和竖直方向居中，宽表/长表仍可完整滚动查看。
6. 检查 Reward 流程：Validity 展开行列和 merge 比对；Structure 展开完整 REF/PRED 1D sequence、首次错误及 `matched/total`；底部展示三项加权算式。
7. 在 Content 卡片确认预测/参考两条分支均展示传入 Anchor OCR 的 PNG、重新解析后的 HTML 与渲染表；两份 Destylized HTML 的 TEDS 直接作为 Content 分量。
8. 点击 Stage 1、Stage 2、SFT、RL，确认仅保留阶段状态且推理按钮禁用，不出现模拟结果。
9. 上传一张 JPEG/PNG/WebP 表格图片并运行；确认预测表格占满结果区并双向居中，保留“提取的 HTML”，不显示参考表格、Reward、Anchor 演示和“模型原始输出”。

## 说明

- Baseline 的行列与 merge 从预测 HTML 派生，仅用于统一展示和 Reward 评估，不是模型独立
  生成的 Sequential Reasoning。
- 代理 Reward 的 Content 使用 Anchor Destylization TEDS；原始 TEDS/TEDS-S 单独展示为 TableVerse 评测指标，不参与加权式，也不要求另启 TEDS 服务。
- Demo 不读取 Phase 1 的预测与分数缓存。100 条 GT 中 99 条具备完整 Reward 的结构前提；
  `tableverse-8ca5eaf2d36eddbb` 的 GT 不通过严格 parser，因此该样例会明确标记 Reward 不可用。
- 曾标为重复退化的两个样本实际是多表原图，快捷入口已经删除；论文所述退化必须以未闭合
  `</table>`、循环内容和命中长度上限等证据确认。
- Demo 对每个 TableVerse 样例显式假设 `should_de_stylize=True`。最终总分是基于 Baseline HTML 派生前置结构的“方法级代理 Reward”，不是已完成 RL 训练后的正式 rollout Reward。
- 首次实时推理才加载 HunyuanOCR；所有 GPU 请求串行执行。
