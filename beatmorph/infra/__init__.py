"""训练基础设施（plan 07）：门禁执行、配置、产物、恢复与环境自检。

| 模块 | 职责 | 里程碑 |
|------|------|--------|
| `sanity.py` | G1-G4 **判据**（范式中立，只吃 step_fn；签名冻结） | M7.2 |
| `gates.py` | 门禁装配 / 落盘 / fail-closed | M7.2 |
| `train_loop.py` | torch 参考训练循环 + 门禁输入装配 + 真实清单批次来源 | M7.2 |
| `lightning_module.py` | Lightning 目标栈（可选依赖，缺失即给可操作报错） | M7.2 |
| `config/` | structured config（OmegaConf 严格合并）与 provenance 必填 | M7.3 |
| `derive.py` | 派生量单一事实源：启动断言 + 全仓字面量扫描 | M7.4 |
| `checkpoint.py` | checkpoint 保存与**恢复校验**（配置 / 门禁 / 数据版本） | M7.5 |
| `feature_cache.py` | 特征缓存元数据的**配置级**帧率校验 | M7.6 |
| `artifacts.py` | 实验目录与六件套契约（不覆盖历史实验） | M7.7 |
| `env_doctor.py` | 环境自检 E1-E5（PASS / FAIL / UNKNOWN 三态） | M7.1 |
| `smoke.py` | 无权重/无网络/无 GPU 的合成批次来源 | M7.2 |

**本包不做**：模型与损失（plan 01/03/04）、评估算法（plan 06）、CLI 用户语义（plan 08）。

导入纪律：本文件**刻意不导入任何子模块**——子模块之间是显式的一对多依赖
（smoke -> data.tracks -> torch、gates -> artifacts -> config），在包入口做汇总导入会把
「只想用 env_doctor」的调用方拖进整条链，并制造循环导入的机会。
"""
