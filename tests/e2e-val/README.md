# `tests/e2e-val/` —— 端到端产物的输入（RFC-0039 R3）

这里放**每 20 000 步生成一张谱面**所需的输入。规则见
[RFC-0039](../docs/decisions/RFC-0039-training-baseline-val-and-e2e-artifacts.md) §R3。

## 目录约定

| 路径 | 入库 | 说明 |
|---|---|---|
| `meta.json` | ✅ | **指针清单**（唯一入库的文件） |
| `audio/` | ❌ | 原始音频（红线 5 附注②：数据不得入库） |
| `feature/` | ❌ | MERT 特征缓存 `.npz` + 元数据 `.json` |
| `outputs/` | ❌ | 生成物 `<YYYYMMDD-HHMMSS>-step<N>/`（本地审阅用） |

## `meta.json` 的字段

```json
{
  "audio": "audio/fudahuang.mp3",        // 必需：只用于 sha1 与来源留痕
  "feature": "feature/fudahuang.npz",    // 必需：MERT 特征（模型输入）
  "feature_meta": "feature/fudahaung.json", // 可选：特征元数据 json
                                              //   （默认取 <feature>.meta.json；
                                              //    本资产的 json 名有拼写差异，故显式指路）
  "chart": null,                          // 可选：模板谱面（RPEJSON）
  "bpm": 120.0,                           // 可选：合成模板的 BPM（默认 120）
  "lines": 16,                            // 可选：合成模板的判定线数（默认 16）
  "difficulty": 15.0,                     // 可选：定数条件（默认 15.0）
  "steps": 8                             // 可选：迭代并行解码步数（默认 8）
}
```

**条件从哪来**（`chart` 缺省时的解析顺序，结果写进产物 `meta.json` 的 `template`）：

1. 按音频内容 **sha1** 在训练清单（`data/processed/pairs.json`）里找回该曲的谱面
   ——取它的判定线事件轨 + BPMList + 定数，这是「条件生成」的原义；
2. 找不到就**合成模板**（`bpm` / `lines`，事件轨全空）⇒ 等价于**无条件生成**，
   一律记为 `template.source = "synthetic"`，与条件生成的产物不可比。

## 产物

`outputs/<时间>-step<N>/`：

- `chart.json`：RPEJSON。**只在合法性后处理后违规项为空时写出**（红线 6）；
- `meta.json`：step / 模板来源与 sha1 / 音频 sha1 / 网格与 τ 轴口径 / 采样器参数 /
  合法性报告与全部统计（**任何一次生成都写**，不合法时它是唯一证据）；
- `notes.txt`：一行摘要。

**它不是门禁**：不产生 PASS/FAIL、不阻塞训练、不参与选权重；失败只告警并继续
（`run.e2e_every` 控制节奏，0 = 关闭）。产物按时间命名，**不覆盖历史**。
