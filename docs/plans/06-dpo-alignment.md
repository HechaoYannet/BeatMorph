# Plan 06 — DPO 偏好对齐

> 状态：🟡 草案 ｜ 阶段：Phase 3（对齐与产品化）｜ 负责：对齐组
> 对应代码：`beatmorph/alignment/dpo.py` ｜ 对应奠基章节：§3.6

## 1. 目标与范围

### 交付
- `DPOTrainer`：以 Stage 2 AR 预训练模型为基座，用 osu! 玩家评分隐式偏好做 DPO 微调，无需独立奖励模型。
- `build_preference_pairs(ratings_db)`：从 osu! 排行榜评分 / PassRate / PlayCount 构造 (Chosen, Rejected) 对。
- `compute_loss(...)`：实现奠基 §3.6.1 目标 `maximize log σ(β · (log π_chosen − log π_rejected))`（含参考模型修正项）。
- 微调循环：策略模型 + 冻结参考模型（基座快照），LoRA 微调以省显存。

### 不交付
- AR 基座本体（Plan 04，仅复用其 checkpoint 作为 policy/ref）。
- 奖励模型训练（奠基 §3.6.2：DPO 不需要，本计划严格不引入）。
- Flow Matching 蒸馏（奠基 §7 Phase 3 另列工程）。
- 数据抓取/清洗本体（归 Plan 08，本计划仅定义 schema 与构造阈值）。

### 价值
用占 RLHF 约 1/3 的成本完成「手感优化」，把社区百万人评分隐式偏好注入模型，直接兜底奠基 R-3「谱面坏习惯」。在 LeVo / MR-FlowDPO 等音乐生成工作中 DPO 已验证可提升音乐性与结构合理性。

## 2. 与奠基文档的对应

| 决策 | 奠基依据 | 本计划 |
|------|---------|--------|
| DPO（非 RLHF/PPO） | §3.6.1 / §8 | 完全采纳 |
| 无独立奖励模型、无需 rollout | §3.6.2 | 完全采纳 |
| 成本约 RLHF 1/3 | §3.6.2 | 采纳，基座 LoRA 冻结主干 |
| 数据 = osu! 评分 / PassRate / PlayCount | §3.6.1 表 / §4.1 | 采纳，schema 见 §4 |
| Chosen = 高评分 ≥4.5 星 & 高 Pass 率；Rejected = 低评分 ≤2 星 | §3.6.1 表 | 完全采纳阈值 |
| 目标 `max log σ(β·(log π_c − log π_r))` | §3.6.1 表 | 采纳，含参考模型项（见偏离） |
| 基座 = Stage2 AR 预训练模型 | §3.6.1 表 | 完全采纳 |
| LeVo/MR-FlowDPO 已验证可提升音乐性 | §3.6.2 | 作为可行性背书，非实现依据 |

**偏离 1**：奠基 §3.6.1 公式仅写 `max log σ(β·(log π_c − log π_r))`，未显式含参考模型项。标准 DPO 完整目标为 `max log σ(β·((log π_c − log π_r) − (log π_ref_c − log π_ref_r)))`，即对比参考模型做相对优化。骨架 `compute_loss` 四参（policy/ref × chosen/rejected）已据此设计。`reference_free=True` 时退化为奠基字面公式（"r-DPO"/reference-free）。记入 RFC-0013。

**偏离 2**：奠基文档未规定 β 值。骨架默认 `beta=0.1`，与音乐符号生成 DPO 常用区间一致；过大破坏多样性，过小对齐无效。记入 RFC-0014。

**偏离 3**：奠基未规定微调方式。本计划规定对 AR 基座做 **LoRA 微调**（冻结主干），参考模型即基座本身，省去第二份全模型显存。记入 RFC-0015。

## 3. 接口契约

对应骨架 `DPOTrainer`（`beatmorph/alignment/dpo.py`）：

```python
class DPOTrainer:
    def __init__(self, beta: float = 0.1, reference_free: bool = False) -> None: ...
    def compute_loss(
        self,
        policy_chosen_logps: Tensor,     # (batch,)  策略模型对 chosen 序列的对数似然
        policy_rejected_logps: Tensor,   # (batch,)
        ref_chosen_logps: Tensor,        # (batch,)  参考模型（reference_free 时忽略）
        ref_rejected_logps: Tensor,      # (batch,)
    ) -> Tensor:                         # () 标量损失
    def build_preference_pairs(self, ratings_db: str) -> list[PreferencePair]: ...
```

张量形状（einops 风格，引用 `core/contracts`）：

| 名称 | 形状 | 含义 |
|------|------|------|
| `*_logps` | `(batch,)` | 单条参考谱面序列的逐 token 对数似然之和 |
| 参考谱面序列 | `(batch, seq_len)` long | `TokenSeq`，值 `∈[0, CODEBOOK_BASE)` |
| AR 条件输入 | `(batch, time_seq, 768)` | `AudioEmbedding`，policy/ref 共用 |
| 偏好对序列长度 | `seq_len ≤ AR_CONTEXT_TOKENS=256` | 受 Plan 04 上下文约束 |
| DPO 条件向量 | 注入 AR 的 difficulty/style | 复用 AdaLN（与 Plan 04 一致） |

- 偏好对 schema：`PreferencePair(chart_id_c, chart_id_r, audio_emb_id, difficulty, chosen_pass_rate, rejected_rating)`，序列从 VQ-VAE 编码对应 `PatternToken` 而来。
- 输出标量 loss 对齐 pydantic / torch 标量契约，供 Lightning 训练循环 `backward`。

## 4. 内部设计

- **偏好对构造**（§3.6.1 阈值）：同一 (audio, 难度档) 下：Chosen = 评分 ≥4.5 星 **且** Pass 率高（≥ P75）；Rejected = 评分 ≤2 星（无视 Pass 率）。同曲不同谱面视为独立样本（奠基 §4.3）。控制变量：仅同 BPM/难度档内配对，避免引入与音频无关的偏好。
- **PlayCount 信号**：作为置信度权重 `w = log10(1+play_count)`，对 loss 加权；PlayCount 极低样本降权，抑制噪声评分。
- **policy/ref 对数似然**：给定 (audio_emb, difficulty, 参考序列 tokens)，复用 Plan 04 AR 前向，取逐 token CE logits → 对数 softmax → 对真实 token 求和得 `logps`。ref = 基座冻结快照（LoRA 关闭即基座本身）。
- **损失**（`compute_loss`）：
  ```
  logits = beta * ((policy_c − policy_r) − (ref_c − ref_r))   # reference_free 时后项=0
  loss = −log_sigmoid(logits).mean()                          # 即最大化 log σ(.)
  ```
  附加：监控 chosen/rejected 准确率与 reward margin `β·(policy_c−policy_r)−β·(ref_c−ref_r)`。
- **为何不用 RLHF（§3.6.2）**：DPO 无需独立奖励模型、无需推理期 rollout，成本约 RLHF 1/3；osu! PlayCount/评分天然提供海量免费偏好数据；LeVo/MR-FlowDPO 已在音乐生成验证可行。
- **训练控制**：早停于 reward margin 不再提升；β 敏感度扫描；防 reward hacking——监控码本坍缩指标（R-2）与生成多样性。

## 5. 依赖关系

- **上游**：Plan 04 AR 预训练 checkpoint（policy/ref 基座）、Plan 02 VQ-VAE（编码参考序列为 `PatternToken`）、Plan 01 音频 emb、Plan 08 数据（osu! 评分库 `ratings_db`）。
- **下游**：Plan 09 CLI/API（暴露微调与对齐后推理）、Plan 04 推理路径（替换基座权重为对齐后权重）。
- **外部库**：`torch>=2.5`、`pytorch-lightning`、`einops`、`peft`（LoRA）、`wandb`。无 `transformers` 强化学习栈（DPO 自实现 loss）。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 3「抓取评分构造 DPO 偏好对微调模型」。

| 里程碑 | 验收 |
|--------|------|
| M1 偏好对构造 | `build_preference_pairs` 产出 ≥50K 对，Chosen/Rejected 阈值 100% 满足 §3.6.1 规则 |
| M2 loss 数值正确 | `compute_loss` 在合成 logps 上与闭式 `−log σ(β·Δ)` 偏差 < 1e-6；ref 增减方向正确 |
| M3 DPO 收敛 | 50K 对微调 reward margin 单调上升后收敛；chosen 准确率收敛于 0.55–0.70 区间（防盗退化） |
| M4 对齐收益盲测 | 对齐后谱面在「手感/可玩性」人工盲测胜率较基座 +10pp 以上；多样性指标下降 < 5% |

## 7. 风险与缓解

| 风险 | 契基编号 | 缓解 |
|------|---------|------|
| osu! 谱面坏习惯被对齐强化 | R-3 | 本模块即 R-3 兜底；仅纳入质量过滤后评分对；chosen 设 Pass 率门槛 |
| 对齐后多样性坍缩 / reward hacking | R-2 | 监控码本覆盖率与 token 熵；多样性下降 >5% 触发回滚；β 调小 |
| 评分布偏（高分段稀少）致 pair 失衡 | — | PlayCount 加权 + 难度档内分层采样；RFC-0016 |
| 参考模型占用双倍显存 | — | LoRA 拓扑使 ref=基座本身，无需双副本；显存仍紧时降 batch（RFC-0015） |

## 8. 测试策略

- **单元**：`compute_loss` 在手造 logps 上数值正确性与方向性（reward margin 正向）；`build_preference_pairs` 阈值/同档配对过滤逻辑；PlayCount 权重计算。
- **集成**：mock AR 前向返回固定 logps，端到端 DPO step 可 backward 且梯度仅作用于 LoRA 参数。
- **e2e**：真实 5K 对 + 真实 AR checkpoint 单卡微调 1 epoch，标注 `@pytest.mark.e2e` `@pytest.mark.gpu`；产出对齐后权重经 Plan 04/07 出可玩 `.osu` 并与基线盲测。

## 9. 开放问题

- [ ] RFC-0013：完整 DPO 目标（含参考项）vs 奠基字面 reference-free 公式的默认选用与 β 等价性。
- [ ] RFC-0014：β 默认值 0.1 的调参区间与选择性扫描协议。
- [ ] RFC-0015：LoRA 微调 ranks / target_modules 与 ref 模型显存预算。
- [ ] RFC-0016：高分段评分稀少时的分层采样与 PlayCount 权重截断策略。
