# future — 板块趋势概率评估系统

从零搭建的板块级趋势概率评估系统。**阶段 1–6 已完成**。

已实现：可复现 Mock 数据、可插拔数据适配器、数据校验、三个结构化因子、拥挤度反向提示、月度分层回测、样本外 IC、新闻同向确认、等权融合、HGB 进阶模型、滚动 IC / PSI 漂移严重度，以及板块概率排序、归因、漂移和回测摘要的 Dash 页面。另提供 `future pipeline` 一键从零生成全部产物。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 调研与选型 | 已完成 |
| 1 | 脚手架 + Mock 数据 + CLI | 已完成 |
| 2 | 三个结构化因子 + 分层回测 | 已完成 |
| 3 | 新闻情感确认 + 等权融合 | 已完成 |
| 4 | sklearn HGB 进阶模型 + 漂移监控 | 已完成 |
| 5 | Dash 可交互页面 | 已完成 |
| 6 | 完整测试 + README 跑通 | 已完成 |

## 一键启动

安装依赖后，推荐直接运行启动脚本：

```bash
cd "E:/python project/future"
uv pip install -e ".[dev,ml,web]"

python start.py
# Windows 也可以双击 start.bat
```

`start.py` 会自动：

1. 检测并切换到项目 `.venv`；
2. 如果页面产物缺失，自动运行完整 `pipeline`；
3. 启动 Dash（默认 `127.0.0.1:8050`）并打开浏览器。

常用参数：

```bash
python start.py --rebuild          # 强制重新生成全部产物
python start.py --port 8051        # 换端口
python start.py --no-browser       # 不自动打开浏览器
python start.py --build-only       # 只生成产物，不启动页面
python start.py --debug            # Dash 调试模式
```

也可以手动分步：

```bash
python -m future.cli pipeline --seed 42
python -m future.cli web --host 127.0.0.1 --port 8050
# 打开 http://127.0.0.1:8050
```

完整质量检查：

```bash
ruff check .
ruff format --check .
pytest -q
```

`pytest` 中包含一条端到端用例：在临时目录从零生成全部产物，再装载页面数据并创建 Dash 应用；不依赖仓库里已有的 `data/` 或 `artifacts/`。

本机验收结果：`ruff` 通过；`pytest` **100 passed**；全量 `pipeline` 生成 37,560 行行情/资金流/新闻、576 行财务、31,296 行因子，并产出回测/新闻/HGB/漂移 artifacts。`python start.py --help` 与 `python start.py --port 8051 --no-browser` 已实测通过。

## 快速开始

```bash
cd "E:/python project/future"

# 1. 创建虚拟环境（Windows Git Bash）
uv venv --python 3.14 .venv
source .venv/Scripts/activate

# 2. 安装项目、开发、ML 与 Web 依赖
uv pip install -e ".[dev,ml,web]"

# 3. 生成并校验 Mock 数据
python -m future.cli generate-data --seed 42 --out data/mock
python -m future.cli validate-data --path data/mock

# 4. 计算结构化因子
python -m future.cli compute-factors \
  --data-dir data/mock \
  --out data/factors \
  --start 2021-01-01 \
  --end 2025-12-31

# 5. 运行样本外分层回测
python -m future.cli backtest \
  --data-dir data/mock \
  --factors data/factors/factors.csv \
  --out artifacts/backtest \
  --oos-start 2024-01-01 \
  --oos-end 2025-12-31

# 6. 新闻同向确认 + 增量分析
python -m future.cli fuse-news \
  --data-dir data/mock \
  --factors data/factors/factors.csv \
  --out data/factors_news \
  --analysis-out artifacts/news \
  --oos-start 2024-01-01 \
  --oos-end 2025-12-31

# 7. HGB 进阶模型
python -m future.cli advanced-model \
  --data-dir data/mock \
  --factors data/factors_news/fused_factors.csv \
  --out artifacts/advanced \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --oos-start 2024-01-01 \
  --oos-end 2025-12-31

# 8. 漂移监控
python -m future.cli drift-monitor \
  --data-dir data/mock \
  --factors data/factors_news/fused_factors.csv \
  --out artifacts/drift \
  --train-start 2021-01-01 \
  --train-end 2023-12-31 \
  --oos-start 2024-01-01 \
  --oos-end 2025-12-31

# 9. 启动 Dash 页面
python -m future.cli web --host 127.0.0.1 --port 8050
# 打开 http://127.0.0.1:8050

# 10. 质量检查
ruff check .
ruff format --check .
pytest -q
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。也可以使用 `python -m venv .venv` + `pip install -e ".[dev,ml]"`。

## 数据设计

默认生成 **24 个板块 × 2020-01-02 至 2025-12-31** 的日频数据。2020 年作为因子预热期，2021–2023 训练、2024–2025 样本外。

| 文件 | 粒度 | 关键字段 | 单位/说明 |
|---|---|---|---|
| `sectors.csv` | 板块 | `sector_id`, `sector_name`, `industry_group` | 板块主数据 |
| `prices.csv` | 板块-日 | `date`, `sector_id`, `open/high/low/close`, `volume`, `amount`, `turnover_rate`, `limit_up_count`, `limit_down_count` | 价格：元；成交量：万股；成交额：亿元；换手率：% |
| `fundamentals.csv` | 板块-季度 | `period_end`, `publish_date`, `pe_ttm`, `pb`, `roe_ttm`, `revenue_yoy`, `profit_yoy`, `debt_ratio` | `publish_date` 晚于 `period_end` 30–60 天，避免前视 |
| `flows.csv` | 板块-日 | `main_net_inflow`, `northbound_net`, `margin_balance`, `large_order_ratio`, `crowding_raw` | 资金流/两融：亿元；拥挤度与主动大单占比：0–1 |
| `news.csv` | 板块-日 | `sentiment`, `news_count`, `positive_count`, `negative_count` | 情感：-1–1；新闻数可为 0 |
| `manifest.json` | 全局 | `seed`, 日期范围, 行数, 文件 SHA-256 | 可复现性证明 |

所有数据写入 `data/`，该目录已被 `.gitignore` 忽略；不提交任何数据文件。

## 结构化因子

因子在**横截面上做 z-score**，日期 t 只使用 t 收盘及之前可用的信息；回测在 t+1 才使用信号。

| 因子 | 定义 | 方向 |
|---|---|---|
| `momentum` | 20/60/120 日收益分别 z-score 后等权 | 正向 |
| `financial` | 最新已披露 `roe_ttm`、`revenue_yoy`、`profit_yoy` 的 z-score 加权（0.4/0.35/0.25） | 正向 |
| `flow` | 20 日主力净流入 + 0.5×北向净流入，除以 20 日成交额，再 z-score | 正向 |
| `crowding` | 拥挤度原始值 5 日平滑后的横截面 z-score | **反向提示** |
| `composite_base` | `mean(momentum, financial, flow) - 0.5 × crowding` | 阶段 2 基线 |
| `composite_fused` | `composite_base + 0.15 × news_confirm` | 阶段 3 融合基线 |

- `financial` 通过 `publish_date <= date` 的 as-of 合并实现，财务尚未披露的日期不会使用未来数据。
- `momentum`、`flow`、`crowding` 都只使用当前及历史窗口；无前视测试覆盖。
- 阶段 3 的融合仍是线性等权基线，不是训练型模型；进阶模型在阶段 4。

## 新闻同向确认

新闻情感**不单独触发**，只作为同向确认条件：

```text
news_count >= 3
且 |sentiment| >= 0.1
且 sign(sentiment) == sign(composite_base)
→ news_confirm = sentiment
→ composite_fused = composite_base + 0.15 × news_confirm
否则 news_confirm = 0，composite_fused = composite_base
```

本机 Mock 覆盖情况：

| 指标 | 结果 |
|---|---:|
| 满足新闻数/情感阈值（可确认） | 38.63% |
| 同向确认占比 | 23.10% |
| 可确认样本中同向比例 | 59.80% |
| 确认样本平均情感 | 0.319 |

相关性分析（全样本 Pearson）：

- `sentiment` 与 `composite_base`：约 0.151（Spearman 约 0.147）
- `news_confirm` 与 `composite_base`：约 0.412（同向筛选带来的机械相关）

融合前后样本外结果：

| 指标 | 基础 composite | 新闻融合 composite | 变化 |
|---|---:|---:|---:|
| 20 日 OOS IC | 0.0689 | **0.0737** | +0.0047 |
| 20 日 ICIR | 0.354 | **0.377** | +0.023 |
| 多空年化收益 | 22.92% | **26.66%** | +3.74pp |
| 多空 Sharpe | 1.85 | **2.10** | +0.25 |
| 多空最大回撤 | -8.86% | **-7.68%** | +1.19pp |

## HGB 进阶模型

- 使用 `sklearn.ensemble.HistGradientBoostingClassifier`（LightGBM/XGBoost 在当前 Python 3.14 下无 Windows cp314 wheel，HGB 是已确认的兜底方案）。
- 标签：t 日之后 20 个交易日收益是否高于当日板块横截面中位数（二分类）。
- 特征：`momentum_20/60/120`、`financial`、`flow`、`crowding`、`sentiment`、`news_count`、`news_confirm`。
- 训练 2021–2023，样本外 2024–2025；训练样本的目标结束日必须不晚于训练期结束，避免标签泄漏。
- 关闭 early stopping，固定 `random_state=42`，保证可复现。
- 线性基线使用 `composite_fused`，与 HGB 使用同一回测口径。

本机 Mock 结果（未计交易成本）：

| 指标 | 线性融合基线 | HGB | 变化 |
|---|---:|---:|---:|
| 5 日 OOS IC | 0.0467 | 0.0401 | -0.0066 |
| 20 日 OOS IC | **0.0737** | 0.0652 | -0.0085 |
| 20 日 ICIR | **0.377** | 0.289 | -0.088 |
| 多空年化收益 | 26.66% | **31.36%** | +4.69pp |
| 多空 Sharpe | 2.10 | **2.46** | +0.36 |
| 多空最大回撤 | -7.68% | **-7.40%** | +0.28pp |
| 多空日胜率 | 56.60% | **58.89%** | +2.29pp |

结论：HGB 在本 Mock 上提高了多空收益与 Sharpe，但 **IC 低于线性融合基线**。它不是无条件的“更强模型”，阶段 5 页面应同时展示两种结果，避免只挑好看指标。

## 漂移监控

参考区间为训练期，监控区间为样本外：

- **滚动 IC**：20 日窗口的横截面 Rank IC；
- **分布漂移**：训练期与最近 20 个交易日窗口的因子 PSI；
- **IC 严重度**：`(训练均值 - 最新滚动 IC) / max(训练标准差, 0.01) / 2`，截断到 `[0, 1]`；
- **分布严重度**：`max(PSI) / 0.5`，截断到 `[0, 1]`；
- **综合严重度**：`0.6 × IC 严重度 + 0.4 × 分布严重度`；
- 等级：`<0.3 low`、`<0.6 medium`、其余 `high`。

本机 Mock 结果：

| 指标 | 结果 |
|---|---:|
| 最新滚动 IC | 0.0340 |
| 训练期 IC 均值 / 标准差 | 0.0851 / 0.2118 |
| IC 严重度 | 0.1205 |
| 分布严重度 | 0.8432 |
| 综合严重度 | **0.4096（medium）** |
| PSI 最大因子 | financial 0.4216 |
| PSI 平均 | 0.1280 |

当前预警主要来自 `financial` 因子的分布漂移，而不是 IC 崩塌；这符合“结构变化先于收益指标恶化”的监控目的。

## 分层回测

- 训练期：2021-01-01 ~ 2023-12-31；样本外：2024-01-01 ~ 2025-12-31。
- 每月最后一个交易日生成信号，**下一交易日**开始持有，避免前视。
- 按 composite 横截面分 5 层，层内等权；同时计算顶层减底层的多空组合。
- 指标：累计/年化收益、年化波动、Sharpe、最大回撤、胜率；样本外 5/20 日 Rank IC、ICIR、IC 胜率。
- 输出未计交易成本，属于可复现的基线口径。

阶段 2 基础 composite 的分层年化（seed=42）：

```text
layer_1: 10.17%   Sharpe 0.50   MDD -35.71%
layer_2: 14.42%   Sharpe 0.63   MDD -36.43%
layer_3: 20.81%   Sharpe 0.81   MDD -34.38%
layer_4: 34.66%   Sharpe 1.24   MDD -34.40%
layer_5: 35.97%   Sharpe 1.27   MDD -36.48%
long_short: 22.92%  Sharpe 1.85  MDD -8.86%
```

## Web 页面

启动：

```bash
python -m future.cli web --host 127.0.0.1 --port 8050
# 或
python -m future.web --host 127.0.0.1 --port 8050
```

页面使用阶段 3–4 的产物：

- 排名：最新 `fused_factors.csv` + HGB `predictions.csv` 的 20 日跑赢概率；
- 点击/选择板块：展示线性因子贡献、近 252 日价格和因子快照；
- 漂移预警：综合严重度 gauge、PSI 因子条、滚动 IC 曲线；
- 回测摘要：线性融合 vs HGB 的 IC、ICIR、多空收益、Sharpe、最大回撤和分层年化。

页面使用顶部切换栏，不再把所有模块堆在同一长页面：

| 页面 | 内容 |
|---|---|
| **日报**（默认） | 一分钟读懂便签、4 个 KPI、大盘板块温度、新闻预测总结 |
| **板块** | 板块概率排序、搜索、点击归因、价格走势 |
| **新闻** | 5 条新闻推送、新闻搜索链接、板块代理展望 |
| **模型健康** | 漂移预警、PSI、滚动 IC、线性 vs HGB 回测对比 |

页面特性：

- 日报页专门总结“板块温度”和“新闻预测”：板块广度、平均相对胜率、头尾差、新闻总量/情感/方向；
- 今日推送新闻卡：近 5 日新闻摘要、外部搜索链接、方向判断与置信度；
- 新闻总结横幅：推送的偏强/中性/偏弱分布、平均情感、最受关注板块和“只做同向确认”的结论；
- 板块展望：新闻 + 财报 + 模型 + 拥挤度的可解释晚融合，支持板块 / 跟踪基金 / 成分股代理模式；
- DataTable 原生排序、分页；顶部提供板块搜索框；点击行或使用选择框选择板块；
- 下拉框搜索板块；
- 归因图为线性 `composite_fused` 分解，HGB 概率单独展示并明确标注不可线性分解；
- 深色数据工作台 + 浅色研究便签的视觉层级；显示字体、等宽小标签和数据表各有角色；
- 无外部 CDN 依赖，页面数据来自本地 artifacts，不调用外部行情服务。

总结模块由 `src/future/web/summary.py` 的 `build_summary()` 和 `build_daily_digest()` 生成，直接读取当前排名、漂移、新闻覆盖和 HGB/线性回测指标；没有金融背景也可以先读结论，再按术语速查理解证据。

## 新闻推送与板块展望

`src/future/web/newsfeed.py` 提供两个输出：

1. **今日推送**：按近 5 个交易日新闻活跃度、情感强度和模型分数挑选板块，生成新闻标题、摘要、方向判断和置信度；标题/底部链接打开东方财富的相关新闻搜索。摘要来自 Mock 数据，不伪造文章 URL。
2. **板块展望**：把四类信号做可解释的晚融合：

```text
方向分 = 50% HGB 相对胜率 + 30% 新闻情感分 + 20% 财报景气分 - 拥挤度扣分
新闻分 = clip(50 + 50 × 近 5 日加权情感, 0, 100)
财报分 = clip(50 + 12 × 财务景气因子 z, 0, 100)
置信度 ≈ 0.35 + 0.3 × 三路信号同向程度 + 0.2 × 新闻覆盖
```

- 当前 Mock 数据只到板块级；页面的“跟踪基金 / ETF（代理）”和“成分股（代理）”明确标注为板块代理，不代表具体证券代码。
- 新闻信号采用 5 日窗口和同向确认计数；新闻不足、情感冲突或拥挤度过高时，判断会更保守。
- 算法参考：FinBERT 情绪 + XGBoost/结构化事件特征、新闻与财务多模态融合、新闻显著性与时间衰减研究。生产化时应接入真实新闻源、FinBERT 类模型和公司级财报，而不是用 Mock 情感替代。

参考：

- [Beyond Sentiment: Structured Information Extraction from Financial News](https://browse-export.arxiv.org/pdf/2607.28496)
- [IKNet: Keyword-Guided News + Technical Indicator Fusion](https://browse-export.arxiv.org/pdf/2510.07661)
- [FinBERT + XGBoost for Event-Driven Stock Prediction](https://www.zenodo.org/records/20011255/files/Vykuntapu_FDAStock_Prediction_2026.pdf?download=1)
- [RavenPack: The Timing Edge — News Sentiment Horizon](https://marketing-prod.ravenpack.com/blog/the-timing-edge-new-research-unlocks-stronger-alpha-from-news-sentiment)
- [Heston & Sinha: News vs. Sentiment — Predicting Stock Returns](https://www.tandfonline.com/doi/abs/10.2469/faj.v73.n3.3)
- [Sentiment-Driven Sector Forecasting](https://ieeexplore.ieee.org/document/11381075)

浏览器验证（Edge 156 / browser-skill 0.4.0）：

- 首页可访问，标题 `future | 板块趋势概率`；
- 6 个 Plotly 图渲染；
- 点击“半导体”行后详情从“计算机”切换为“半导体”，概率从 66.6% 切到 56.1%；
- 下拉选择“电力”后详情切换为 55.6%；
- 板块列筛选“半导体”后仅剩 1 行；
- 控制台无错误。

## 产物与可复现性

```text
data/
├─ mock/             # 行情、财务、资金流、新闻、manifest
├─ factors/          # factors.csv + manifest.json
└─ factors_news/     # fused_factors.csv + manifest.json
artifacts/
├─ backtest/         # metrics.json、layer_returns.csv、ic_series.csv、manifest.json
├─ news/             # analysis.json、manifest.json
├─ advanced/         # predictions.csv、metrics.json、manifest.json
└─ drift/            # rolling_ic.csv、drift.json、manifest.json
```

- `data/` 和 `artifacts/` 都被 `.gitignore` 忽略。
- 数据、因子、新闻融合、HGB 预测和漂移文件均使用稳定列顺序、`\n` 换行、6 位小数和 `YYYY-MM-DD` 日期格式。
- 同 seed / 同参数重复运行生成相同的文件哈希；所有 manifest 记录 SHA-256。
- `validate-data` 会校验数据字段、取值域和 manifest 哈希。
- 因子无前视测试会改动 cutoff 之后的行情/资金流/财务，验证 cutoff 之前的因子完全不变。
- 新闻规则测试覆盖阈值不足、方向相反、基础信号为 0 等“不确认”场景，确保新闻不单独触发。
- HGB 训练标签额外限制 `target_end <= train_end`；测试断言 `train_label_end` 不超过 2023-12-31。

## 数据适配器

```text
DataAdapter (Protocol)
├── MockDataAdapter      默认，内存生成，无网络依赖
├── CsvDataAdapter       读取已生成的本地 CSV
└── FallbackAdapter      包装 RealDataAdapter；失败自动降级到 Mock
```

- `FUTURE_DATA_SOURCE=mock|csv|real` 选择数据源，默认 `mock`。
- `RealDataAdapter` 目前是占位实现，调用 `load()` 会抛 `DataSourceError`；`info` 会提示将降级到 Mock，实际加载失败时自动回退。
- 数据源适配器不读取、不保存密钥值；真实接入时只通过供应商 SDK 的环境变量机制取凭据。

## CLI

```bash
# 数据
python -m future.cli generate-data --seed 42 --out data/mock --json
python -m future.cli validate-data --path data/mock --json
python -m future.cli info --json

# 因子与回测
python -m future.cli compute-factors --data-dir data/mock --out data/factors --json
python -m future.cli backtest \
  --data-dir data/mock \
  --factors data/factors/factors.csv \
  --out artifacts/backtest \
  --json

# 新闻同向确认
python -m future.cli fuse-news \
  --data-dir data/mock \
  --factors data/factors/factors.csv \
  --out data/factors_news \
  --analysis-out artifacts/news \
  --min-news-count 3 \
  --min-abs-sentiment 0.1 \
  --boost 0.15 \
  --json

# HGB 进阶模型
python -m future.cli advanced-model \
  --data-dir data/mock \
  --factors data/factors_news/fused_factors.csv \
  --out artifacts/advanced \
  --json

# 漂移监控
python -m future.cli drift-monitor \
  --data-dir data/mock \
  --factors data/factors_news/fused_factors.csv \
  --out artifacts/drift \
  --json

# 一键阶段 1–4
python -m future.cli pipeline --seed 42 --json

# Web 页面
python -m future.cli web --host 127.0.0.1 --port 8050
```

配置也可通过环境变量覆盖：

```bash
FUTURE_SEED=7 FUTURE_DATA_SOURCE=mock FUTURE_DATA_DIR=data/mock \
  python -m future.cli info
```

## 校验规则

- 必要字段齐全，日期/数值类型正确，无空值。
- `sector_id` 唯一且所有表引用一致。
- 日频表 `(sector_id, date)` 唯一，日期升序。
- `low <= min(open, close) <= max(open, close) <= high`，价格、成交量、成交额非负。
- 新闻情感在 `[-1, 1]`，且 `positive_count + negative_count <= news_count`。
- 财务披露 `publish_date > period_end`，防止前视。
- 拥挤度、主动大单占比在 `[0, 1]`。
- manifest 存在且各 CSV 的 SHA-256 匹配。

## 目录结构

```text
future/
├─ pyproject.toml
├─ README.md
├─ start.py                  # 跨平台启动脚本
├─ start.bat                 # Windows 双击入口
├─ .gitignore
├─ src/future/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ config.py
│  ├─ cli.py
│  ├─ evaluation.py       # 前瞻收益、目标结束日、日度 Rank IC
│  ├─ factors.py          # 三个结构化因子 + 拥挤度 + composite
│  ├─ backtest.py         # 月度分层回测 + OOS IC
│  ├─ news.py             # 新闻同向确认 + 增量分析
│  ├─ ml.py               # HGB 进阶模型 + 线性对比
│  ├─ drift.py            # 滚动 IC / PSI / 0–1 严重度
│  ├─ pipeline.py         # 一键阶段 1–4 流水线
│  ├─ start.py            # 启动脚本核心逻辑
│  ├─ web/
│  │  ├─ data.py          # 页面数据装载
│  │  ├─ summary.py       # 通俗结论、阅读提示与术语速查
│  │  ├─ newsfeed.py      # 新闻推送、板块展望与代理标的说明
│  │  ├─ app.py           # Dash 布局、回调和图表
│  │  └─ assets/style.css # 深色工作台 + 晨报便签视觉系统
│  └─ data/
│     ├─ base.py          # Dataset / DataAdapter / FallbackAdapter
│     ├─ mock.py          # 确定性 Mock 生成
│     ├─ storage.py       # CSV + manifest
│     ├─ registry.py      # mock/csv/real 解析与降级
│     └─ validate.py      # 数据校验
└─ tests/
```

## 已知限制

- 回测未计交易成本、滑点和借券成本；HGB 与新闻融合的增益都需要扣费复核。
- HGB 没有做超参数搜索；当前只是可复现的基线模型，IC 反而低于线性融合。
- 新闻阈值（3 条、0.1、0.15）和漂移权重（0.6/0.4、PSI/0.5）是启发式参数，尚未做敏感性分析。
- Mock 使用 `pd.bdate_range` 近似交易日，未建模 A 股节假日。
- Web 页面是本地 Dash 开发服务器，未实现登录、权限、生产 WSGI 部署和并发缓存；从 `http://127.0.0.1:8050` 访问。
- 页面数据依赖阶段 1–4 的本地产物；未生成 artifacts 时页面会给出缺失文件提示。重新生成产物后需要重启 Web 服务，当前没有自动热刷新。
- DataTable 在 Dash 未来大版本中可能移除；`pyproject.toml` 已固定 `dash>=4.5,<5`，当前 Dash 4.5 下已验证。升级到 Dash 5 前应评估迁移到 `dash-ag-grid`。
- 当前环境 Python 3.14；LightGBM/XGBoost 无 Windows cp314 wheel，故使用 sklearn HGB。
- 项目已初始化 git，阶段 1–6 归档在首个提交；`data/`、`artifacts/`、`.venv/` 不进入版本控制。

## 依赖安装

- 已完成：`uv pip install -e ".[dev,ml,web]"`，包含 Dash 4.5.0、Plotly 7.1.0、scikit-learn 1.9.1。
- 如需单独安装 Web：`uv pip install -e ".[web]"`。
