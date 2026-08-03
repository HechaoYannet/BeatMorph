# BeatMorph 训练操作手册

> 本文档是 Phase 1 「环境 → 数据下载 → 预处理 → MERT 离线提取 → Stage1 训练」的**端到端操作手册**,
> 并说明各产物如何供 VQ-VAE(Plan 02)与 AR Transformer(Plan 04)开发衔接。
> 设计依据见 [`BasePlan.md`](BasePlan.md) §4/§5/§7 与各 [plans/](plans/README.md);数据获取见 [RFC-0024](decisions/RFC-0024-sayobot-data-source.md)。

---

## 0. 前置约定

- 目标设备:**GPU 设备**(Stage1 训练需 CUDA;MERT 提取 FP16 显著加速)。CPU 仅可做数据解析/小规模冒烟。
- 操作系统:Linux 优先(GPU 训练);Windows 亦可(uv 命令一致,PowerShell 语法略异)。
- 包管理:统一 [`uv`](https://docs.astral.sh/uv/),**禁止直接 pip**(保证 `uv.lock` 可复现)。
- 红线:模型权重/原始音频/全量数据集**不入库**(见 [`CLAUDE.md`](../CLAUDE.md) §3 红线 5),走 Git LFS 或各设备自行提取。

---

## 1. 环境准备

```bash
# 1.1 clone(含 LFS,拉取 data/dev_sample 小型冒烟样本)
git clone https://github.com/HechaoYannet/BeatMorph.git BeatMorph
cd BeatMorph
git lfs install        # 首次启用 LFS(若未全局启用)
git lfs pull           # 拉取 LFS 大文件(当前仅 data/dev_sample/charts.parquet)

# 1.2 安装依赖(GPU 训练全量组)
uv sync --extra audio --extra train --extra rag --extra data --group dev
```

**依赖组说明**(`pyproject.toml [project.optional-dependencies]`):

| extra | 含 | 何时需要 |
|-------|----|---------|
| `audio` | transformers / **modelscope** / peft / soundfile / librosa | MERT 编码器、音频加载(Stage0/提取) |
| `train` | pytorch-lightning / wandb / pyarrow / datasets | Stage1 训练、Parquet 读写 |
| `rag` | faiss-cpu | Phase2 RAG(Phase1 可不装) |
| `data` | httpx | `scripts/download_sayobot.py` 数据下载 |
| `dev`(默认组) | ruff / pytest / mypy / pre-commit | 开发/测试 |

> **CUDA PyTorch**:`pyproject.toml [tool.uv]` 已配 `pytorch-cu130` index,`uv sync` 自动选 CUDA 轮子。
> 若设备 CUDA 版本不匹配 13.0,按 [PyTorch 官网](https://pytorch.org/) 指引调整 index。

### 环境变量

```bash
export BEATMORPH_RAW_DIR=data/raw           # 原始 .osu+音频 落地根(默认)
export BEATMORPH_DATA_DIR=data/processed    # 预处理产物 + embeddings 根(默认)
export BEATMORPH_RUNS_DIR=runs              # 训练产出(ckpt/log)根(默认)
# MERT 权重 cache(二选一,见 §2):
export HF_HOME=$HOME/.cache/huggingface     # HuggingFace cache
# 或 ModelScope cache(国内推荐):export MODELSCOPE_CACHE=$HOME/.cache/modelscope
```

### 冒烟验证

```bash
uv run pre-commit install        # 提交钩子
make test-fast                   # 快测试(跳过 slow/gpu/e2e):应全绿
uv run ruff check . && uv run mypy beatmorph   # lint + 类型:应 0 error
```

> `tests/unit/audio/test_mert.py` 与 `tests/integration/test_audio_to_plan.py` 标 `@pytest.mark.gpu`,
> 且会在 MERT 权重未缓存时 **skip**(不失败)。权重就位后(见 §2)自动解除 skip。

---

## 2. MERT 权重获取

MERT-v1-330M(~1.3GB)从 **ModelScope → HuggingFace fallback** 拉取(`configs/model/mert.yaml: source: modelscope`)。

```bash
# 方式 A:ModelScope(国内可达性最好,默认)
uv run python -c "from modelscope import snapshot_download; print(snapshot_download('m-a-p/MERT-v1-330M'))"

# 方式 B:HuggingFace(海外)
uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)"
```

权重落到 `$HF_HOME` 或 `$MODELSCOPE_CACHE`。`MERTAdapter` 构造时按 `source` 优先级自动解析本地缓存,**不再联网**(除非 cache 缺失)。

**验证权重就位**(应返回 True,且随后 MERT 测试不再 skip):

```bash
uv run python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True) is not None)"
uv run pytest tests/unit/audio/test_mert.py -q     # 权重+GPU 就位后不再 skip
```

---

## 3. 真实数据下载(sayobot.cn 镜像,RFC-0024)

```bash
# 列表模式:按 offset 翻页抓 osu!mania 4K 谱面集(默认 --only-4k --skip-unranked)
uv run python scripts/download_sayobot.py \
    --list-mode \
    --max-sets 5000 \
    --raw-dir data/raw \
    --rate-delay 1.0

# 或关键字搜索
uv run python scripts/download_sayobot.py --keyword "felys" --max-sets 50
```

**产物**(`$BEATMORPH_RAW_DIR`,均 gitignored):
- `data/raw/{sid}/*.osu` + 引用的音频(`audio.mp3`/`.ogg`)
- `data/raw/manifest.jsonl`:每行一个 set 的集级元数据(`sid/stars/play_count/approved/...`),供 §4 激活质量过滤

**关键参数**(对照 [`configs/data/download.yaml`](../configs/data/download.yaml)):
- `--only-4k` / `--no-only-4k`:仅保留含 4K mania diff 的 set(奠基 §1.1 4K 优先)
- `--skip-unranked` / `--no-skip-unranked`:跳过未 Ranked/Pending(`order=0` 或 `approved=3`,§4.3 本就会剔,此为省带宽)
- `--rate-delay`:每 set 间隔秒(礼貌限流,默认 1.0)
- `--retries`:连接级重试(默认 3)
- `--no-verify`:`cmcc.sayobot.cn:25225` 证书链 httpx 默认验不过,脚本遇 `SSLCertVerificationError` 自动降级 `verify=False` 并 warn;`--no-verify` 全程不校验

**吞吐建议**:10K set 单线程 + 1s 限流约 3-4 小时(含下载+解压)。sayobot 为非官方镜像,
若遇 429/封禁,增大 `--rate-delay` 或换时段。断点续传:重跑同命令,已下载的 `{sid}/` 自动跳过。

> TOS/版权:训练数据仅学术研究用途(BasePlan §4.3)。下载产物不入库(红线 5)。

---

## 4. 数据预处理(PreprocessPipeline,§4.2 Step1-2)

```bash
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline

pipe = PreprocessPipeline(
    raw_dir=Path('data/raw'),
    out_dir=Path('data/processed'),
    manifest_path=Path('data/raw/manifest.jsonl'),   # 传入才激活 §4.3 质量过滤
)
pipe.run()           # 解析 .osu → Chart + 清理 + §4.3 过滤 + compute_section_stats 伪标签
pipe.write_stats()   # 写 data/processed/stats.json
"
```

**产物**(`$BEATMORPH_DATA_DIR`):
- `charts.parquet`(主)+ `charts.jsonl`(pyarrow 缺失时兜底):每行一个 Chart IR,含 `notes` / `bpm_points` / `sections`(伪标签) / `meta`(含注入的 `difficulty_rating`/`playcount`/`audio_duration`)
- `stats.json`:`{total, parsed, passed_filter, skipped_mode, failed}`

**发生了什么**:
1. `parse_osu`:`OsuManiaReader` 解析 `.osu` → `Chart`;多编码兜底;负时间/越界 lane Note 清理。
2. **manifest 注入**(RFC-0024):按 `beatmap_set_id` 把集级 `difficulty_rating`/`playcount` 注入 `Chart.meta`(只加不覆盖)。
3. **§4.3 质量过滤**:传入 manifest 后,`≥3 星 且 play_count > 500` 真正生效;未传入则走「Phase 1 relaxed pass」(宽松通过,用于无元数据的 fixture 测试)。
4. **音频时长注入**:`soundfile.info` 读音频全长 → `meta["audio_duration"]`,使 Section 边界与推理同源。
5. `compute_section_stats`:按 `bpm_points` 切每 4 小节一个 `Section`,回填 `density_target/energy_level/rest_probability/sections_type`(Stage1 伪标签,零标注)。
6. **mode 诚实标记**(RFC-0025):非 4K mania 标真实 K 数落盘,**不降级 4K**;4K 主路径靠训练层 `mode==MANIA_4K` 过滤守 R-6(见 §6 `PlannerDataset`)。

> 多 K 谱面(5/6/7/8K)会在此步骤落盘到 `charts.parquet`,但 `PlannerDataset` 训练时会过滤掉,只吃 4K。

---

## 5. MERT 离线提取(§4.2 Step3)

```bash
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline

pipe = PreprocessPipeline(raw_dir=Path('data/raw'), out_dir=Path('data/processed'))
pipe.extract_mert_embeddings(
    audio_dir=Path('data/raw'),       # rglob *.osu,按 .osu 父目录解析 chart.audio_path
    source='modelscope',              # 权重来源优先级
    device='auto',                    # CUDA 可用则用
    limit=None,                       # None=全量
)
"
```

**产物**:`$BEATMORPH_DATA_DIR/embeddings/mert_v1_330m/{beatmap_set_id}.pt`(按 set 去冗余,同 set 多难度共享一份),每个 `[T_seq, 1024]` float32
(`T_seq ≈ duration_s × 25`)。`adapter='none'`(纯冻结主干表征,离线提取不训 Adapter)、FP16、**断点续抽**(已存在 `.pt` 跳过)。

**校验**:

```bash
uv run python -c "
import torch, glob
for p in sorted(glob.glob('data/processed/embeddings/mert_v1_330m/*.pt'))[:3]:
    e = torch.load(p, map_location='cpu', weights_only=True)
    print(p, tuple(e.shape), e.dtype)   # 期望 (T_seq, 768) float32
"
```

**耗时/显存**:单曲 T4 FP16 约 0.3-0.5s(单曲);10K 约 1-2 小时。长音频按 5s 窗/1s 重叠滑窗 + 重叠区平均。离线一次产出多次训练复用(§4.2「节省训练时算力」)。

> 若跳过本步,`PlannerDataset` 会因缺 embedding 把该样本 skip(`n_missing` 计数入日志)。

---

## 6. Stage1 密度规划训练(奠基 §7 Phase1)

```bash
# Hydra 驱动,配置见 configs/stage1_planner.yaml
uv run python -m beatmorph.cli.train --config-name stage1_planner

# 常用覆盖:
uv run python -m beatmorph.cli.train --config-name stage1_planner \
    experiment.max_steps=50000 \
    train.batch_size=4 \
    trainer.devices=1
```

**配置**(见 [`configs/stage1_planner.yaml`](../configs/stage1_planner.yaml) + [`configs/model/planner.yaml`](../configs/model/planner.yaml)):
- 数据:`data.charts_path` + `data.embeddings_dir`(默认 `$BEATMORPH_DATA_DIR/charts.jsonl` 与 `.../embeddings/mert_v1_330m`)
- 模型:6 层双向 Transformer,`dim=768`(与 MERT 对齐免投影),3 回归头 + 5 类段落类型头
- 训练:`bf16-mixed` + 梯度裁剪 1.0 + AdamW(lr 3e-4);`Huber×3 + TV + CE(type)` 多任务损失
- ckpt:`$BEATMORPH_RUNS_DIR/checkpoints`,每 2000 步存 top-1

**数据集**(`PlannerDataset`,自动配对 charts + embeddings):
- 读 `charts.jsonl`,每行 `Chart`;**跳过 `mode != MANIA_4K`**(守 R-6,日志记 `n_non_4k`)
- 跳过无 sections / 无对应 `.pt` 的样本(日志记 `n_missing`)
- 每样本 yield:`audio_emb [T_seq,768]` + `target{density,energy,rest,type}` + `difficulty` + `section_bounds [S+1]`

**数据集空怎么办**:`train.py` 会 `raise RuntimeError("PlannerDataset 空...")`。原因通常是:
- `charts_path` 指错(应是 `data/processed/charts.jsonl` 而非 `.parquet`——**注意**:训练读 jsonl,`PreprocessPipeline` 默认产 parquet,需设 `cache_format='jsonl'` 或手动转换)
- embeddings 未提取(先跑 §5)
- 全是 non-4K(检查下载 `--only-4k`)

> **重要**:`PlannerDataset` 读 **`charts.jsonl`**(`Chart.model_validate_json` 逐行反序列化),
> 而 `PreprocessPipeline` 默认写 `charts.parquet`。生产流程请让 pipeline 写 jsonl
> (`cache_format='jsonl'`),或用 pyarrow 读 parquet 转 jsonl。dev_sample 提供了 parquet 样式
> 供 VQ-VAE/AR 参考(见 §7)。

**W&B**:Phase1 默认 `logger=False`(`build_trainer`),Phase2 接入(见 `infra.trainer`)。

---

## 7. 产物与下一步衔接(VQ-VAE / AR 开发)

### 各 Stage 产物清单

| 产物 | 路径 | 由谁产出 | 下游消费者 |
|------|------|---------|-----------|
| `charts.{parquet,jsonl}` | `data/processed/` | §4 PreprocessPipeline | VQ-VAE 训练(Plan02)、AR 训练(Plan04)、PlannerDataset |
| MERT embeddings `.pt` | `data/embeddings/mert_v1_330m/{beatmap_set_id}.pt`(按 set 去冗余) | §5 extract_mert | PlannerDataset、AR Cross-Attention |
| Stage1 planner ckpt | `runs/checkpoints/` | §6 train.py | AR 规划条件(Plan04) |
| MERT+Adapter ckpt | (Phase1 可选,`adapter='none'` 离线提取不含 Adapter) | `--config-name stage0_mert` | Stage0 推理 |

### VQ-VAE(Plan 02)开发需要什么

- **输入**:`Chart`(含 `notes`/`bpm_points`),来自 `charts.jsonl` 或 `data/dev_sample/charts.parquet`
- **栅格化**:按 `bpm_points` 分段推小节边界(RFC-0005),把 Note 投到 `(lane, time_bins)` 网格
- **冒烟样本**:`data/dev_sample/charts.parquet`(2 首 fixture 4K Chart,走 LFS),用法见 [`data/dev_sample/README.md`](../data/dev_sample/README.md)
- **验收目标**:重建准确率 > 95%(Plan02 M2)、`codebook_usage() ≥ 0.5`(R-2)

### AR Transformer(Plan 04)开发需要什么

- **输入**:VQ-VAE encode 的 `TokenSeq` + audio_emb + Stage1 planner 的 `list[Section]` + RAG 前缀(Phase2)
- **依赖**:VQ-VAE tokenizer 训练完成(Plan02)、Stage1 planner ckpt(§6 产出)
- **上下文**:256 tokens(`AR_CONTEXT_TOKENS`)

### dev_sample 用法(开发期冒烟,无需跑全量)

```bash
git lfs pull    # 拿 data/dev_sample/charts.parquet
uv run python -c "
import pyarrow.parquet as pq
from beatmorph.core.contracts import Chart
rows = pq.read_table('data/dev_sample/charts.parquet').to_pylist()
charts = [Chart.model_validate_json(r['chart_json']) for r in rows]
print(len(charts), 'charts for VQ-VAE/AR shape smoke')
"
```

> dev_sample 仅 2 首 fixture,够验形状/往返,**不可训练**。embedding 需 GPU 设备补齐(见 `data/dev_sample/README.md`)。

### 全量产物在设备间传递

**不通过 git**——全量 charts/embeddings/ckpt 体量数十 GB,超 GitHub LFS 免费 1GB 配额。
各 GPU 设备按 §3-§6 自行下载→处理→提取→训练。LFS 仅承载 `data/dev_sample` 微样本。
若需跨设备传训练好的 ckpt,用外部对象存储(OSS/S3)或 `rsync`/`scp`,见 RFC-0023(开放)。

---

## 8. 故障排查

| 现象 | 原因 / 解决 |
|------|------------|
| `test_mert.py` 全 skip | MERT 权重未缓存或无 CUDA。跑 §2 拉权重,确认 `AutoConfig.from_pretrained` 可解析 |
| `PlannerDataset 空` | 见 §6「数据集空怎么办」;常见为 charts_path 指向 parquet(应 jsonl)、embeddings 未提取、全 non-4K |
| 质量过滤「Phase 1 relaxed pass」反复出现 | 未传 `manifest_path` 或 manifest 不含对应 `beatmap_set_id`。确认 §3 产 manifest 且 §4 传入了 `manifest_path` |
| sayobot `SSL: CERTIFICATE_VERIFY_FAILED` | 脚本自动降级 `verify=False`;或显式 `--no-verify` |
| sayobot 429/连接被拒 | 增大 `--rate-delay`(如 2.0-3.0),换时段,或减少并发(脚本单线程) |
| `compute_section_stats` 产出 0 sections | `bpm_points` 为空(检查 `.osu` TimingPoints 有非继承红线)或 `notes` 为空;pipeline 会兜底 `bpm=120` |
| `extract_mert_embeddings` 大量 `no audio` skip | `chart.audio_path`(basename)在 `.osu` 父目录找不到音频;确认 §3 解压完整(音频与 .osu 同 set 目录) |
| mypy `unused section: module = ['pyarrow.*']` | 良性告警——当前环境未装 pyarrow 时报;`uv sync --extra train` 后消失 |
| LFS `smudge error` / clone 后 parquet 是指针 | 未跑 `git lfs pull`;或 LFS 配额耗尽(检查 GitHub 账户 LFS 用量) |

---

## 9. 一期训练最小可跑序列(cheatsheet)

```bash
# 0. 环境
git clone https://github.com/HechaoYannet/BeatMorph.git BeatMorph && cd BeatMorph
git lfs pull
uv sync --extra audio --extra train --extra data --group dev
export BEATMORPH_RAW_DIR=data/raw BEATMORPH_DATA_DIR=data/processed BEATMORPH_RUNS_DIR=runs

# 1. MERT 权重(一次性)
uv run python -c "from modelscope import snapshot_download; snapshot_download('m-a-p/MERT-v1-330M')"

# 2. 下载数据(先小规模验证:50 set)
uv run python scripts/download_sayobot.py --list-mode --max-sets 50 --rate-delay 1.0

# 3. 预处理(产 charts + sections;cache_format 设 jsonl 供训练读)
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline
p = PreprocessPipeline(Path('data/raw'), Path('data/processed'), cache_format='jsonl', manifest_path=Path('data/raw/manifest.jsonl'))
p.run(); p.write_stats()
"

# 4. MERT 离线提取
uv run python -c "
from pathlib import Path
from beatmorph.data.pipeline.embed import PreprocessPipeline
PreprocessPipeline(Path('data/raw'), Path('data/processed')).extract_mert_embeddings(Path('data/raw'))
"

# 5. Stage1 训练
uv run python -m beatmorph.cli.train --config-name stage1_planner experiment.max_steps=10000

# 6. 验收:三关绿 + 训练 loss 下降
make test-fast
```

> 一期先小规模(50 set)跑通全链路,再扩到 10K(set `--max-sets 10000` + `experiment.max_steps=100000`)。

---

## 相关文档

- [BasePlan.md](BasePlan.md) §4(数据)/ §5(基础设施)/ §7(路线图)— 技术权威
- [plans/01-audio-encoder.md](plans/01-audio-encoder.md) — MERT 编码器
- [plans/03-planner-density.md](plans/03-planner-density.md) — Stage1 密度规划
- [plans/08-data-pipeline.md](plans/08-data-pipeline.md) — 数据流水线
- [plans/09-infra-cli-api.md](plans/09-infra-cli-api.md) — 训练基础设施
- [decisions/RFC-0024](decisions/RFC-0024-sayobot-data-source.md) — sayobot 数据源
- [decisions/RFC-0003](decisions/RFC-0003-adapter-lora-vs-mlp.md) — MERT LoRA Adapter
- [decisions/RFC-0005](decisions/RFC-0005-bpm-timepoints.md) — 变速 bpm_points
- [decisions/RFC-0025](decisions/RFC-0025-multikey-data-r6.md) — 多 K 数据与 4K 训练过滤
