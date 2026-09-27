# RFC-0032 — 局部层的事件轨条件注入改为「本线轨」

- 状态：**采纳**（2026-09-27 由决策者裁定：局部层只看本线运动，跨线只发生在全局层）｜ 提出日期：2026-09-27 ｜ 决定日期：2026-09-27
- 提出者：主会话（训练基础设施 / 模型架构）
- 影响模块：plan 04（§4.1/§4.2 的解码器条件注入与局部/全局分工）、plan 07（显存墙与 K 上限）、
  `beatmorph/generation/model.py`；**不动** plan 00/03 的契约

## 背景

本轮（2026-09-27 第六轮）为回答「显存墙到底在哪里」做了一组**有守卫**的单步实测
（`scripts/local_vram_wall.py`：不强制注意力后端、每步前后查 `mem_get_info`、按 `(K·T)²` 外推超预算即停）。
生产配置（B=1、d_model=256、heads=4、n_layers=6、window=32、global_period=4、`t_window=192`、
`x_bins=128`、fp32）上：

- 峰值显存 `≈ 1.05 + 0.00391·K² GiB`、步时 `≈ 0.00054·K² s`（**K=32 是独立校核点**：预测 5.05 / 实测 5.05）；
- ⇒ 本机 8 GB 卡的安全上限 **K ≈ 28-31、物理上限 ≈ 38**，而 train split **36.4% 的谱面 K > 31、
  27.3% 的 K ≥ 40** ⇒ **按当前配置直接开全量训练，会在最初几个样本上就 OOM**。

**第一个反直觉事实：墙不在全局层。** 把 6 层全部原地翻成局部层（改 `DecoderLayer.kind`，
**参数完全相同**）在 K=20 时峰值 **2.92 GiB**，全部翻成全局层只有 **1.22 GiB**——
与「global 层对 `K·T` 做全自注意力 ⇒ `O((K·T)²)` 是墙」的既有叙述**正好相反**。

**第二个对照定位了机制**：把 `line_tracks` 的时间轴从 192 格压到 **1 格**（只改 K/V 长度、纯输入侧），
全局部层 **2.92 → 1.29 GiB**、步时 **1.169 → 0.064 s（18×）**；生产排布 2.47 → 1.26 GiB。
⇒ 局部层的代价几乎全部来自**轨道 cross-attention，其 key 轴长度是 `K · T_line`**。

代码事实（`MaskedFieldModel.decode` / `encode_tracks`）：

```python
tracks = self.encode_tracks(batch)          # (B, K * T_line, d)：K 条线拍平成一条序列
...
local = module(flat, cond.repeat_interleave(K), audio.repeat_interleave(K),
               tracks.repeat_interleave(K),  # (B*K, K*T_line, d)
               attn_mask=band_mask)
```

`tracks` 的**序列维里已经含判定线轴**，再按线 `repeat_interleave` 之后，**每条线的 token 都
attend 到全部 K 条线的事件轨**——在 4 个局部层，无任何 mask，连 τ 也不切窗。这与模块 docstring
「局部层是滑动窗口自注意力 / 全局层（层内同时看到全部 K 条线的场）」在**判定线轴上直接冲突**。
注意滑动窗口本身是**正确**的：它作用在 `self_attn` 上，按线（batch = B·K）分、τ 上只开 ±window。

**顺带回答「能不能上 flash attention」：不能解决这个墙。** 能力查询显示这些形状
`mem_efficient=True`（本机 torch 2.13 + sm_120，fp32 下 `flash=False`），但真换上
`F.scaled_dot_product_attention` 只省 **8%**（K=20：2.92 → 2.69 GiB）——因为 K/V 的**重复发生在
注意力之前**（batch 维就是判定线数），任何核都消不掉 `K² · T_line · d` 的投影与驻留。

## 提议

**局部层的事件轨条件只注入本线的事件轨；跨线信息只经全局层。**

实现：`encode_tracks` 产出的 `(B, K*T_line, d)` 在局部层 reshape 成 `(B*K, T_line, d)`——
每条线的 query 只看自己那条线的 `T_line` 个轨 token。**全局层不变**（继续用未拆分的 `tracks`
作 K/V），因此 RFC-0029 §2.4-4 的「全局层同时看到全部 K 条线」仍然成立。

### 实测后果

| K | 改前（全库轨） | 改后（本线轨） |
|---|---|---|
| 20 | 2.46 GiB | 1.31 GiB |
| 32 | 5.05 GiB | 1.86 GiB |
| 48 | **OOM** | 2.74 GiB |
| 64 | OOM | 3.41 GiB |
| 96 | OOM | 3.93 GiB（+ bf16） |
| **128** | OOM | **5.10 GiB（剩 1.70 GiB，+ bf16）** |

- 显存从 **∝K² 变成 ∝K**（1.31 / 1.86 / 2.74 / 3.41 对 K=20/32/48/64 基本是一条直线）；
- 本线轨 + bf16（`optim.precision: bf16-mixed`，见 plan 07 §9-36）**覆盖到 `k_max = 128`**，
  即**当前语料全部谱面，一个都不用丢**。

## 备选方案

1. **保留全库轨、只建一份 K/V**（把 `cross_tracks` 从按线的 batch 里提出来、像全局层那样一次算完）：
   语义**完全不变**、显存也回到 ∝K，但**计算仍是 K²**，且要为「共享 K/V + 分组 query」另造一条批处理路径。
   **保留为回归路径**：若训练后发现局部层确实需要别线的轨，这条是代价最小的回退。
2. **换 flash / mem_efficient 核**：实测只省 8%，因为重复在注意力之前。不解决问题。
3. **维持现状 + 数据侧 K 截断**：要丢掉 36.4% 的谱面（含 39.7% 的有效音符）才能跑起来。
   这正是本轮要避免的事——**数据完整性优先于任何 harness**。

## 后果

- plan 04 §4.2 的局部/全局分工要写清「跨线只发生在全局层」；`model.py` 的模块 docstring 已同步。
- plan 07 §9-35 的结论改写：墙的根因是**条件注入的 K/V 作用域**，既不是全局层、也不是注意力核。
- K 上限不再是训练的前置阻碍，`data.k_max = 128` 重新成为真正的上限来源。
- **契约级护栏**：`tests/unit/generation/test_local_own_line_tracks.py` 直接钉住 cross_tracks 的
  K/V 长度（局部 = `T_line`、全局 = `K*T_line`）。这条护栏是必须的——改回旧写法**不会报任何错**，
  只会让显存重新变成 O(K²)。
- 局部层能看到的条件变少，G1-G4 需要在真机上重跑一轮（随 `--gates` 启动时**顺带**完成，不额外占 GPU）。

## 本 RFC **不**决定的两件事

1. **装饰线（不承载任何有效 note 的判定线，占全库判定线的 57.8%）是否从 λ 的线轴里去掉。**
   决策者已定：**用额外的旁路完成，标记为后续扩展，等主路线训练完成后再做**
   （见 plan 04 §9 与 plan 07 §9-37）。本 RFC 只保证「在保留全部 K 条线的前提下，训练能跑起来」。
2. **无条件生成（只给音频）时装饰线由谁产出**——随第 1 条一起留到后续扩展。

## 关联

- 实测脚本与原始记录：`scripts/local_vram_wall.py`（§A 阶梯 / §E 层型 / §H 轨长度 / §I SDPA / §J 本线轨 / §K 组合）、
  `runs/_vram_wall/records.jsonl`
- plan 04 §4.1/§4.2、plan 07 §9-35/§9-36、`docs/TRAINING.md` §7.5
- [RFC-0029](RFC-0029-phigros-continuous-chart-generation.md) §2.4-4（全局层同时看到全部 K 条线）
- [phira-dataset-survey.md §7.7](../knowledges/phira-dataset-survey.md)（装饰线占比的语料级实测）
