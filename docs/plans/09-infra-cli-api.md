# Plan 09 — 训练/推理基础设施与 CLI/API

> 状态：🟡 草案 ｜ 阶段：Phase 1-3 ｜ 负责：基础设施组
> 对应代码：`beatmorph/infra/`、`beatmorph/cli/`、`beatmorph/api/`、`configs/` ｜ 对应奠基章节：§5、§7

## 1. 目标与范围

### 交付
- Hydra 配置体系（`configs/{model,train,data}`），组合式超参管理。
- PyTorch Lightning 训练框架封装（FSDP 多卡 A100）。
- W&B 实验管理（损失/生成样例/BPE 词表利用率）。
- 检查点与可复现性（`uv.lock` + 种子控制 + 配置快照）。
- CLI：`beatmorph-generate`（端到端生成）；`beatmorph-train`（训练各 Stage）。
- API：推理服务（Phase 3 Web Demo）。

### 不交付
- 模型本体（各 Stage 在 Plan 01-06）、后处理（Plan 07）、数据预处理（Plan 08）。

### 价值
奠定文档 §5 基础设施规格的落地：统一训练/推理入口，保证多人跨设备环境一致与实验可复现。

## 2. 与奠基文档的对应

| 决策 | 奠基依据 | 本计划 |
|------|---------|--------|
| Python 3.10+ | §5 | 采纳，固定 3.11 |
| uv 包管理 | §5 | 采纳，`pyproject.toml` + `uv.lock` |
| PyTorch 2.5+ + torch.compile | §5 | 采纳 |
| PyTorch Lightning / FSDP | §5 | 采纳，多卡 A100 |
| W&B 实验管理 | §5 | 采纳 |
| Hydra 配置 | §5 | 采纳，`configs/` 分层 |
| 训练 4×A100(80G) / 推理 1×T4(16G) | §5 | 采纳为推荐规格 |
| ONNX Runtime / TensorRT | §5 | 标记可选，Phase 3 落地 |

**偏离**：奠基未指定日志方案。本计划统一 `beatmorph.core.logging`，支持 JSON 输出（容器化）。记入 RFC-0022。

## 3. 接口契约

### CLI（`pyproject.toml [project.scripts]`，待启用）
```bash
beatmorph-generate --audio song.mp3 --difficulty 8 --ref reference.osu -o out.osu
beatmorph-train --config-name stage2_ar
```
对应 `beatmorph/cli/generate.py::main()`。

### 配置（Hydra）
```
configs/
├── model/        ar_transformer.yaml, bpe.yaml, planner.yaml, mert.yaml
├── train/        stage2_ar.yaml, stage1_planner.yaml, stage0_bpe.yaml
└── data/         osu_50k.yaml, osu_1m.yaml
```
> RFC-0028 后 tokenizer 改 BPE/event：`model/vqvae.yaml`、`stage_vqvae.yaml`、`_run_vqvae` 分发属 VQ-VAE baseline，将随 `archive/vqvae-baseline` 分支切走，主路径不再维护；新 BPE 训练对应 `model/bpe.yaml` / `train/stage0_bpe.yaml`（`ar_transformer.yaml` 已用 `vocab_size=4096`，RFC-0028）。
环境变量约定：`BEATMORPH_DATA_DIR`、`BEATMORPH_RUNS_DIR`、`BEATMORPH_CACHE_DIR`。

### API
`beatmorph/api/app.py::create_app()`，输入音频+难度+参考谱面，返回 `.osu`。Phase 3 实现。

## 4. 内部设计

- **infra/config**：Hydra ConfigGroup，OmegaConf 解析；运行时快照配置到 `runs/<exp>/config.yaml` 保证可复现。
- **训练循环**：LightningModule 封装各 Stage；支持 `bf16-mixed`、梯度累积、梯度裁剪、FSDP 策略。
- **实验管理**：W&B 记录 loss / lr / BPE 词表利用率（R-2 随 VQ 退役，词表利用率仅作多样性参照）/ 生成样例（每 N 步解码一张 `.osu` 预览）。
- **检查点**：按 step 存档 + 保留 top-K；从 ckpt 续训接口。
- **可复现**：`uv sync` 重建环境；`seed=42`；配置 + 代码 git tag 锁定。

## 5. 依赖关系

- **上游**：所有 Stage 模块、数据流水线。
- **下游**：用户（CLI/API）。
- **外部库**：`hydra-core`、`omegaconf`、`pytorch-lightning`、`wandb`、`rich`。

## 6. 里程碑与验收标准

| 里程碑 | 验收（对应奠基 §7） |
|--------|------|
| M1 环境一键复现 | 全新机器 `uv sync && make test` 通过（Phase 1） |
| M2 Stage0/1/2 训练脚本 | 三 Stage 可用 CLI 训练、W&B 上线（Phase 1-2） |
| M3 端到端 CLI 生成 | `beatmorph-generate` 输出合法可玩 `.osu`（Phase 2 末） |
| M4 推理 API | 封闭测试 Web Demo 可用（Phase 3） |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| 多卡/多设备环境漂移 | — | `uv.lock` + `.python-version` + `.gitattributes(LF)` 强制一致 |
| 实验不可复现 | — | 配置快照 + 种子 + 代码 tag；W&B 关联 commit |
| 推理延迟（API 场景） | R-4 | Phase 3 Flow Matching 蒸馏 + TensorRT |

## 8. 测试策略

- **单元**：Hydra 配置可加载、CLI 参数解析、ckpt 续训 shape 一致。
- **集成**：`make test` 在干净环境冒烟通过。
- **e2e**：`@pytest.mark.e2e` 跑 `beatmorph-generate`，校验输出 `.osu` 可被 Plan 07 校验为合法。

## 9. 开放问题

- [ ] RFC-0022：日志 JSON 化是否对所有 Stage 强制。
- [ ] RFC-0023：检查点存储用本地 vs 云（OSS/S3）。
- [ ] API 框架选型（FastAPI vs Litestar）待 Phase 3 定。
