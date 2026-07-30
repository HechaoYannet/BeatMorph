# models/

预训练与训练产物权重存放目录。

- `pretrained/`：MERT-v1-330M 等预训练权重（运行期由代码下载填充，**禁止入库**，见根 `.gitignore`）。
- 训练检查点默认输出到 `$BEATMORPH_RUNS_DIR`（见 `.env.example`），不在本目录。

本目录仅保留结构占位，不存放任何权重大文件。
