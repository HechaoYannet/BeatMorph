# RFC-0003 — MERT Adapter 选型：LoRA vs MLP

- 状态：采纳 ｜ 提出日期：2026-07-31 ｜ 决定日期：2026-07-31
- 提出者：音频组
- 影响模块：plan 01（`beatmorph/audio/encoder/mert.py`）、plan 02/04（消费 `audio_emb` 的训 Adapter 影响）、`pyproject.toml`（新增 `peft`）

## 背景

奠基 §3.1.1 规定 MERT-v1-330M **冻结主干、仅训轻量 Adapter**（R-1 缓解：曲风适配不足时回退微调后 6 层）。Plan 01 §4 同时提及两种 Adapter 注入方式：

1. **LoRA**：对 MERT 后若干层 attention 投影注入低秩矩阵（rank=8），主干 `requires_grad_(False)`。
2. **MLP**：在指定层输出后接 2 层 MLP 残差 Adapter。

二者性能均可达曲风适配目标，但实现复杂度、显存、可调参数量、迁移到 Demucs 分轨的复用性不同。Plan 01 §9 把默认值列为开放问题（RFC-0003 提案）。现据 Phase 1 起步需求定型。

## 提议

**采纳 LoRA 为默认 Adapter**（`MERTAdapter(adapter="lora")`），MLP 保留为备选（`adapter="mlp"`）。

- 实现：`peft.LoraConfig` 注入 MERT attention 的 `query`/`value` 投影，`rank=8`、`alpha=16`、`dropout=0.05`；`peft.get_peft_model` 包装冻结主干。主干全部 `requires_grad_(False)`，仅 LoRA 参数 trainable。
- `adapter="none"`：纯冻结直出（不做曲风适配），用于离线 embedding 预提取等只需主干表征的场景（plan 08 Step3）。
- `beatmorph/audio/encoder/mert.py` 构造签名补 `adapter: str = "lora"`、`lora_rank: int = 8`（plan 01 §3.1 已预期）。

## 备选方案

1. **MLP Adapter**：在 `layer=12` 输出后接 2 层 MLP 残差。实现更简单（纯 `nn.Linear`+GeLU）、调试直观。但：① 不改变主干 attention 内部计算，曲风适配力弱于 LoRA 对 attention 的直接调制；② 残差 MLP 参数量随 `feat=768` 线性，相对 LoRA 的 rank=8 低秩更费显存；③ 复用 HF `peft` 生态的成熟度/迁移性差。**保留为备选**（LoRA 不达标或显存极受限时回退）。放弃作为默认。
2. **全量微调 MERT 后 6 层**：奠基 §3.1 / R-1 提及的"严重时回退"方案。参数量过大（>100M），不符"轻量 Adapter"范式，且 330M 全微调对 Phase 1 数据量（10K-50K）易过拟合。仅在 R-1 显化时考虑。放弃。
3. **不加 Adapter（纯冻结 MERT）**：`adapter="none"`。零训练成本，但放弃曲风适配能力，偏离奠基"冻结+Adapter"范式。仅限离线提取表征等只读场景。不作为训练默认。

## 后果

- **依赖新增 `peft>=0.11`**（`audio` extra），`MERTAdapter` 改为 `torch.nn.Module` 子类（建立仓库首个 torch 模型实现先例）。
- **可训参数**：仅 LoRA 矩阵（~rank×2×768×层数级，远 <1M），主干 330M 冻结 grad=0；FP16/BF16 训练显存友好。
- **plan 01 §9**：RFC-0003 标记「采纳」，plan 文本已写 `adapter="lora"` / `rank=8`，无需改 plan；M1/M2 验收（冻结参数断言、Adapter 可训参数集断言）以本 RFC 口径。
- **下游影响**：MERT 输出 `[B,T_seq,768]` 不变，Stage1 planner 与 Stage2 AR 直连免投影（plan 04 §2）。Adapter 只改 encoder 内部，输出契约不破。
- **风险**：若 MERT-v1-330M 的 attention 模块命名与 `peft` 默认 `target_modules=["query","value"]` 不完全匹配，需按实际模块名调整（实现时用 `peft` 的 module introspection 确认，记入 plan 01 §9）。

## 关联

- 奠基：§3.1.1（冻结主干 + Adapter）、§3.1（R-1 曲风适配）、§8 决策矩阵。
- plan 01 §3.1（接口契约 `adapter` 参数）、§4（Adapter 注入）、§9（RFC-0003 开放问题）。
- 相关 RFC：无直接链；权重加载来源见 RFC-0024 的 source 优先级思路（ModelScope→HF）。
