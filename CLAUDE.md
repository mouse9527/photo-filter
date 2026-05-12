# CLAUDE.md

本仓库是一个 LLM 驱动的照片筛选工具，用于扫描 Sony A7C2、DJI Action4 等照片目录，分析废片候选，并提供 Web Review 页面辅助人工复核。

## 项目结构

- `src/photo_filter/cli.py`：Click CLI 入口，包含扫描、分析、移动、统计、Web 启动等命令。
- `src/photo_filter/web.py`：FastAPI Review Web UI。
- `src/photo_filter/analyzer.py`：调用 LLM 判断照片质量。
- `src/photo_filter/scanner.py`：扫描照片文件。
- `src/photo_filter/mover.py`：移动或 dry-run 处理照片文件。
- `src/photo_filter/db.py`：PostgreSQL/SQLAlchemy 数据库访问。
- `src/photo_filter/static/`：前端静态页面。
- `tests/`：pytest 测试。

## 常用命令

- 安装依赖：`uv sync --extra dev`
- 运行测试：`uv run pytest`
- 运行 lint：`uv run ruff check .`
- 查看 CLI：`uv run photo-filter --help`
- 启动 Review Web：`uv run photo-filter --config config.yaml web`

## 工作要求

- 沟通和 PR/issue 回复优先使用中文。
- 优先做小而明确的改动，不要为了当前问题引入不必要的抽象或重构。
- 修改行为前先理解现有 CLI、Web、数据库、文件移动流程之间的关系。
- 涉及真实照片文件、NFS 路径、删除/移动文件、数据库迁移、Kubernetes/CronJob、生产配置时必须谨慎；不要在没有明确要求时执行破坏性操作。
- 不要提交 API key、数据库密码、LiteLLM token、真实照片路径或其他敏感信息。
- `config.example.yaml` 只能放示例值，真实配置应留在本地或部署环境。
- 提交前尽量运行与改动相关的最小验证；通常至少运行 `uv run pytest` 或 `uv run ruff check .`。

## Claude Code GitHub Actions

- GitHub Actions 中的 Claude 只应响应显式 `@claude` 请求。
- 不要自动合并 PR，不要绕过 CI、pre-commit 或 review 流程。
- 如果修改文件移动、删除、数据库或部署相关逻辑，应在回复中说明风险和建议的人工验证步骤。
