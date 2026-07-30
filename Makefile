# BeatMorph — 常用命令快捷方式
# 用法: make <target>   （需先安装 uv: https://docs.astral.sh/uv/）
# Windows 下推荐在 PowerShell 直接用 uv 原生命令；Makefile 供类 Unix 环境使用。

.PHONY: help install dev lint format typecheck test test-fast clean git-init

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 通过 uv 安装项目（含运行时依赖）
	uv sync --extra audio --extra train --extra rag

dev: ## 安装开发环境（含 dev 依赖）
	uv sync --group dev

lint: ## Ruff 代码检查
	uv run ruff check .

format: ## Ruff 自动格式化
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## mypy 类型检查
	uv run mypy beatmorph

test: ## 运行全部测试
	uv run pytest

test-fast: ## 仅快测试（跳过 slow/gpu/e2e）
	uv run pytest -m "not slow and not gpu and not e2e" -p no:cacheprovider

clean: ## 清理产物
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml dist build *.egg-info
	rm -rf runs outputs wandb mlruns 2>/dev/null || true

git-init: ## 安装 pre-commit hook
	uv run pre-commit install
