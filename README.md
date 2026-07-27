# Sci Host MCP

Sci Host MCP 是一个面向材料科学文献调研的 Model Context Protocol 服务。它把论文检索、质量筛选、CSP 知识抽取、隐性文献配对、Research Gap 识别、可证伪假设生成、多次仿真验证和结构化报告串成一个可审计工作流。

这个仓库只包含 Sci Host MCP 服务本身。它不依赖、也不包含其他上级项目、数据集或本机路径。

## 能做什么

- 从离线语料、arXiv、OpenAlex 或 Sciverse 获取论文。
- 使用质量闸门过滤缺少技术机制、可观测证据或目标方向命中的论文。
- 抽取 `Composition → Structure → Property` 三元组，并保留论文来源信息。
- 计算论文之间的语义配对，寻找跨材料体系的方法迁移和隐藏连接。
- 生成带材料、结构、工艺、性能和证据线索的可证伪候选假设。
- 在独立科研仿真运行时中执行多算子一致性检查、参数扰动重跑和交叉验证。
- 将每一轮的论文、配对、假设、试错、验证和 Agent 反馈记录为可回溯状态。
- 通过 MCP 暴露批处理工具、Agent CSP 工具、流式推理工具和竞赛报告工具。

系统输出的是候选科学假设和证据链，不是已经通过真实实验、DFT、MD 或商业数据库确认的科学结论。正式结论必须由后续计算或实验验证。

## 架构

```text
论文源
  -> PaperCrawler
  -> ResearchQualityGate
  -> CSP Extractor
  -> ImplicitPairer
  -> HypothesisGenerator
  -> ResearchTwin 仿真与 TrialEngine
  -> VerificationEngine 多次复现
  -> KnowledgeAccumulator / 比赛报告 / MCP
```

`ResearchTwin` 是本仓库内的轻量运行时，负责论文语料状态、评估算子、试错记录和在线校准。它不要求安装其他数字孪生项目。

## 安装

要求 Python 3.10 或更高版本（MCP 依赖的当前版本要求）。

```bash
git clone <repository-url>
cd sci-host-mcp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[mcp,dev]"
```

只运行离线 Python 流程时可以省略 `[mcp]`。

## 快速运行

离线材料科学单轮运行：

```bash
python -m sci_host.run --materials --single-step
```

离线多轮运行：

```bash
python -m sci_host.run --mode materials --iterations 5 --interval 1
```

打开 Demo：

```text
demo/competition-demo.html
```

Demo 是静态展示页，不会伪造 MCP 调用。真实接入时应使用 MCP 客户端连接服务并查看返回的论文、证据和验证记录。

## 配置 Sciverse

不要把 token 写进代码、README、MCP JSON 或 Git 提交。使用环境变量：

```bash
export SCIVERSE_API_TOKEN="your-token"
```

随后在 MCP 客户端中创建材料 Host，关键参数如下：

```json
{
  "host_id": "materials-demo",
  "offline": false,
  "source": "sciverse",
  "materials_mode": true,
  "quality_gate": true,
  "keywords": ["perovskite", "thermoelectric", "solid electrolyte"]
}
```

如果客户端通过 stdio 启动服务，应把 `SCIVERSE_API_TOKEN` 放在客户端的 `env` 字段中，而不是放入仓库文件。完整配置示例见 [MCP_CONFIG.md](MCP_CONFIG.md)。

## MCP 服务

stdio 模式：

```bash
python -m sci_host.mcp_server
```

HTTP 模式：

```bash
python -m sci_host.mcp_server --http 8080
```

也可以使用安装后的命令：

```bash
sci-host-mcp
```

### 推荐比赛演示流程

1. `sci_create_materials_host` 创建材料模式 Host。
2. `sci_stream_crawl` 采集论文并让 Agent 评估关注重点。
3. `sci_stream_pair` 查看跨论文配对并选择值得追踪的连接。
4. `sci_stream_hypothesize` 生成构效关系或工艺假设。
5. `sci_stream_trial` 查看每个评估算子的预测和失败原因。
6. `sci_stream_verify` 运行多次复现并输出候选等级。
7. `sci_get_discovery_report` 或 `sci_generate_competition_report` 导出结构化结果。

流式反馈工具 `sci_stream_feedback` 可以记录 Agent 的 reasoning、关注论文、关注配对、关注假设以及 force pass/force fail。Agent 覆盖会单独标记，不会伪装成正式验证发现。

## 主要工具分组

- Host 管理：`sci_create_host`、`sci_create_materials_host`、`sci_get_info`、`sci_list_hosts`、`sci_remove_host`。
- 探索循环：`sci_step`、`sci_run_cycles`、`sci_get_status`、`sci_get_snapshot`。
- 知识查询：`sci_get_pairs`、`sci_get_csp_knowledge`、`sci_get_top_validated`、`sci_get_failures`、`sci_get_graph`。
- 基础任务：`sci_get_gap_evidence`、`sci_get_literature_review`、`sci_generate_competition_report`。
- Agent CSP：`sci_get_papers_for_csp`、`sci_submit_csp`、`sci_step_agent`。
- 流式推理：`sci_stream_crawl`、`sci_stream_pair`、`sci_stream_hypothesize`、`sci_stream_trial`、`sci_stream_verify`、`sci_stream_feedback`。
- 在线数据源：`sci_sciverse_search`、`sci_sciverse_status`。

## 测试

```bash
python -m pytest -q
```

测试覆盖质量闸门、CSP 数值抽取、流式阶段、竞赛工具、物理范围校验、端到端报告和离线主流程。真实 Sciverse 测试需要显式设置 token，默认测试不会读取或输出 token。

## 发布边界

- `.env`、运行日志、缓存、原始 JSONL、备份文件和本机生成数据默认被忽略。
- 仓库不保存任何 API token、模型密钥或本机绝对路径。
- 离线模式适合演示和回归测试；在线模式的论文结果取决于当前数据源响应。
- 内部仿真只用于候选排序与可复现性筛选，不能替代实验室验证。
