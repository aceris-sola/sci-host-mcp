# MCP 客户端配置

Sci Host MCP 是一个本地 stdio 服务。MCP 客户端负责启动进程，服务负责维护研究运行状态并响应工具调用。

## 基础配置

安装依赖后，推荐通过 Python 模块启动：

```json
{
  "mcpServers": {
    "sci-host": {
      "command": "python3",
      "args": ["-m", "sci_host.mcp_server"],
      "cwd": "/path/to/sci-host-mcp",
      "env": {
        "SCIVERSE_API_TOKEN": "由本机环境注入"
      }
    }
  }
}
```

也可以使用安装后的命令：

```json
{
  "mcpServers": {
    "sci-host": {
      "command": "sci-host-mcp",
      "env": {
        "SCIVERSE_API_TOKEN": "由本机环境注入"
      }
    }
  }
}
```

支持 stdio 的 MCP 客户端包括 Claude Desktop、Cursor、VS Code Continue 等。配置后重启客户端，先调用 `sci_get_info` 检查服务能力，再创建研究运行。

## 选择数据源

默认创建的 Host 可以使用离线语料，适合验证安装：

```json
{
  "tool": "sci_create_materials_host",
  "arguments": {
    "host_id": "materials-local",
    "offline": true,
    "batch_size": 20
  }
}
```

如果要连接 Sciverse，先在启动 MCP 进程的环境中设置 Token，再调用 `sci_create_sciverse_host`：

```bash
export SCIVERSE_API_TOKEN="your-token"
```

不要把真实 Token 写进配置文件并提交到 Git。客户端是否支持从系统环境变量展开，取决于客户端实现；不确定时，直接在启动 MCP 进程的环境中设置变量。

## 推荐调用顺序

### 自动运行

调用 `sci_step` 执行一轮完整研究循环：

```text
论文采集 -> 质量筛选 -> CSP 抽取 -> 跨论文配对
-> 候选假设 -> ResearchTwin 试错 -> 知识积累
```

### Agent 逐阶段运行

如果希望模型读取每一阶段的结果后再决定下一步，使用：

```text
sci_stream_crawl
-> sci_stream_pair
-> sci_stream_hypothesize
-> sci_stream_trial
-> sci_stream_verify
```

阶段之间可以调用 `sci_stream_feedback` 指定关注的论文、配对或假设。反馈会作为运行记录保存；如果使用 force pass 或 force fail，结果会带有 Agent 覆盖标记。

## 检查真实数据源

每次在线运行都应检查返回对象中的：

- `data_source`：`online`、`offline_fallback` 或 `mixed`；
- 论文的 `source`、论文 ID 和原始链接；
- 质量闸门的接受/拒绝数量；
- 验证结果中的 `credibility` 和 `credibility_note`。

在线请求失败时，服务可能回退到内置语料。回退结果适合保持流程可用，但不能当作真实数据库检索结果引用。

## 安全检查

不要将 `SCIVERSE_API_TOKEN`、LLM key 或外部数据库 key 写入 JSON、代码或 Git。使用客户端的进程环境、操作系统密钥环或 CI secret。发布前执行：

```bash
git grep -n -I -E 'github_pat_|sk-[A-Za-z0-9_-]{20,}|sci_[A-Za-z0-9]{20,}'
```

命令没有输出才表示仓库文本中没有匹配到这些常见 Token 格式；它不能替代平台侧的 secret scanning。
