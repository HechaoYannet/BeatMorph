# `data/dev_sample/` — VQ-VAE / AR 开发冒烟样本

> 走 Git LFS(`*.parquet` 规则)。clone 后 `git lfs pull` 拉取。

## 这是什么

一份**小型、版权安全**的 Chart IR 样本,供 VQ-VAE(Plan 02)与 AR Transformer(Plan 04)在
开发期做**形状/流程冒烟**(编解码往返、Token 序列构建、 DataLoader 通路),无需先跑全量数据流水线。

## 当前内容

| 文件 | 说明 |
|------|------|
| `charts.parquet` | 2 首 4K mania Chart IR(含 `sections` 伪标签 + `bpm_points` + `notes`),每行一个 `chart_json` 列 + 统计列 |

**规模诚实声明**:当前仅 2 首,源自 `tests/fixtures/sample_4k_mania.osu` +
`sample_minimal_mania.osu`(仓库内置微型 fixture,非真实谱面)。足够验证 pipeline 形状与
往返逻辑,**不足以训练**。VQ-VAE/AR 真实训练需按 [`docs/TRAINING.md`](../../docs/TRAINING.md)
跑全量 `PreprocessPipeline` 产出。

样本由 `PreprocessPipeline(raw=tests/fixtures, out=data/dev_sample).run()` 生成;
非 mania fixture(如 `sample_taiko.osu`)会被自动 skip。

## 列 schema(`charts.parquet`)

| 列 | 类型 | 说明 |
|----|------|------|
| `chart_json` | string | `Chart.model_dump_json()`,完整 IR(可 `Chart.model_validate_json` 反序列化) |
| `title` / `artist` | string | 元数据 |
| `difficulty` | int64 | 1-15 |
| `mode` | int64 | GameMode(`MANIA_4K=4`) |
| `bpm` | double | `chart.primary_bpm()` |
| `num_notes` / `num_sections` / `num_bpm_points` | int64 | 统计 |

## 缺什么(embedding)

本样本**不含 MERT embedding `.pt`**——embedding 需 MERT 权重才能生成,仓库不做权重入库。
VQ-VAE/AR 冒烟若需 embedding,在 GPU 设备首次 clone 后自行补齐:

```bash
# 用本样本的 .osu(或自带音频)跑离线提取,产出 data/embeddings/mert_v1_330m/{bid}.pt
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline
PreprocessPipeline(Path('tests/fixtures'), Path('data/dev_sample')).extract_mert_embeddings(Path('tests/fixtures'))
"
```

> fixtures 的 `AudioFilename: audio.mp3` 实际不存在,提取会 skip。真实冒烟请用带音频的
> 谱面(按 `docs/TRAINING.md` 下载数据后提取)。

## VQ-VAE / AR 冒烟用法(开发期)

```python
import pyarrow.parquet as pq
from beatmorph.core.contracts import Chart

rows = pq.read_table("data/dev_sample/charts.parquet").to_pylist()
charts = [Chart.model_validate_json(r["chart_json"]) for r in rows]
# → 喂给 VQVAETokenizer.encode(chart) / ARTransformer 占位通路做形状验证
```

## 不要扩样本入库

**禁止**把真实谱面/音频/全量 embedding 入库(CLAUDE.md 红线 5)。本目录只放微小 fixture 产物。
全量数据在各设备按 `docs/TRAINING.md` 自行获取,不入 git。
