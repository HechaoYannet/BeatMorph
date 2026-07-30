# 代码结构详解

> 本文件描述 BeatMorph 仓库的物理结构与模块职责，便于贡献者快速定位。
> 技术决策见 [`BasePlan.md`](BasePlan.md)，实施计划见 [`plans/`](plans/README.md)。

## 顶层布局

```
BeatMorph/
├── beatmorph/            # 主 Python 包
├── tests/                # 测试（unit / integration / e2e / fixtures）
├── configs/              # Hydra 配置（model / train / data）
├── docs/                 # 文档（BasePlan / plans / decisions / 本文件等）
├── scripts/              # 一次性运维/数据脚本
├── data/                 # 数据（git 忽略，仅 fixtures 入库）
├── models/               # 模型权重（git 忽略，走 LFS/外部存储）
├── pyproject.toml        # 项目与工具配置（uv/ruff/pytest/mypy/coverage）
├── Makefile              # 常用命令快捷方式
├── CLAUDE.md             # AI 协作宪法
├── AGENTS.md             # 子 agent 协作约定
└── LICENSE               # Apache-2.0
```

## 包结构（beatmorph/）

```
beatmorph/
├── __init__.py                # 版本号 + 模块拓扑说明
├── core/
│   ├── contracts/             # ★ 跨模块数据契约（最高频引用）
│   │   ├── events.py          #   Note/Chart/Section/PatternToken/NoteType/GameMode
│   │   └── tensors.py         #   张量形状约定 + 常量（25Hz/2048/256 等）
│   └── logging.py             #   统一日志 get_logger()
├── audio/
│   ├── encoder/mert.py        # Stage 0  MERTAdapter: wav -> [B,T,768]
│   └── separation/demucs.py   # 可选     DemucsSeparator: 四轨分离
├── tokenizer/vqvae.py         # VQ-VAE  encode/decode/codebook_usage
├── planner/density.py         # Stage 1 DensityPlanner.plan -> list[Section]
├── generation/ar_transformer.py # Stage 2 ARTransformer.generate -> list[PatternToken]
├── rag/retriever.py           # RAG     build_index / retrieve
├── alignment/dpo.py           # DPO     compute_loss / build_preference_pairs
├── decoder/
│   └── postprocess/constraints.py # Stage4 PostProcessor.apply/validate（物理红线）
├── io/formats/
│   ├── base.py                # ChartWriter / ChartReader 抽象基类
│   ├── osu.py                 # .osu (osu!mania) 读写
│   └── sm.py                  # .sm (StepMania) 读写
├── data/
│   ├── parsers/osu_path.py    # parse_osu / compute_section_stats
│   └── pipeline/embed.py      # PreprocessPipeline（4 步流水线）
├── infra/config/              # Hydra 封装
├── cli/generate.py            # beatmorph-generate 入口
└── api/app.py                 # 推理服务（Phase 3）
```

## 数据流（端到端，对应 BasePlan §2）

```
用户输入: audio(wav/mp3) + difficulty(1-15) + 参考谱面(可选)
   │
   ├─[audio.encoder] MERT+Adapter ───────────────► audio_emb [B,T,768] @25Hz
   │              └(可选) Demucs 四轨分离增强
   ├─[rag.retriever]  检索 Top-K=3 参考谱面 ─────► rag_prefix
   ├─[planner.density] 自监督回归 ────────────────► sections (每4小节 density/energy/rest)
   ├─[generation.ar]   AR Transformer ──────────► PatternToken 序列 [B,≤256]
   ├─[tokenizer.vqvae] decode ──────────────────► Chart IR (Note[]: time,lane,type,duration)
   ├─[decoder.postprocess] 物理红线校验 ────────► 合法 Chart
   └─[io.formats.osu]  write ───────────────────► .osu (可玩谱面)
```

## 测试结构（tests/）

```
tests/
├── conftest.py            # 共享 fixtures
├── unit/                  # 与 beatmorph/ 镜像，每模块子目录
│   └── core/test_contracts.py   # ★ 契约冒烟测试（可立即运行）
├── integration/           # 多模块串联（Stage0→1→2 等）
├── e2e/                   # 音频→谱面端到端
└── fixtures/              # 小型合法样本（.osu/.mp3 片段，可入库）
```

测试标记：`slow` / `gpu` / `integration` / `e2e`（见 `pyproject.toml [tool.pytest]`）。

## 配置结构（configs/）

`configs/{model,train,data}/` 为 Hydra 配置组，组合式超参：
```bash
uv run python -m beatmorph.cli.train --config-name=stage2_ar \
    model=ar_transformer data=osu_50k
```
环境变量：`BEATMORPH_DATA_DIR` / `BEATMORPH_RUNS_DIR` / `BEATMORPH_CACHE_DIR`。

## 现状

- ✅ 工程脚手架、git、契约、日志、配置样例、测试样例就绪。
- ⬜ 各模块为接口占位（`NotImplementedError` + 指向 plan），待按 Phase 1-4（BasePlan §7）逐步实现。
