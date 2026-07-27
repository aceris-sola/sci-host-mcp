# MCP 配置

安装依赖后，推荐通过 Python 模块以 stdio 方式启动：

```json
{
  "mcpServers": {
    "sci-host": {
      "command": "python3",
      "args": ["-m", "sci_host.mcp_server"],
      "cwd": "/path/to/sci-host-mcp",
      "env": {
        "SCIVERSE_API_TOKEN": "由本机环境注入，不要提交真实值"
      }
    }
  }
}
```

如果已经执行 `pip install -e ".[mcp]"`，也可以直接使用：

```json
{
  "mcpServers": {
    "sci-host": {
      "command": "sci-host-mcp",
      "env": {
        "SCIVERSE_API_TOKEN": "由本机环境注入，不要提交真实值"
      }
    }
  }
}
```

支持 stdio 的 MCP 客户端包括 Claude Desktop、Cursor、VS Code Continue 等。配置后重启客户端，并先调用 `sci_get_info` 检查服务能力，再创建材料模式 Host。

## 安全配置原则

不要将 `SCIVERSE_API_TOKEN`、LLM key 或外部数据库 key 写入 JSON、代码或 Git。使用客户端的进程环境、操作系统密钥环或 CI secret。发布前执行：

```bash
git grep -n -I -E 'github_pat_|sk-[A-Za-z0-9_-]{20,}|sci_[A-Za-z0-9]{20,}'
```

命令没有输出才表示仓库文本中没有匹配到这些常见 token 格式；它不能替代平台侧的 secret scanning。
