# 极简 LLM：从 0 到 1 搭建微型 GPT 并内化近体诗格律

> 作者：WING · 2026-09-14
> 定位：纯 Python + numpy，从 0 到 1 手写微型 GPT 的训练与采样链路，并在此之上用
> 平水韵格律（句长 / 押韵 / 平仄 / 对仗）约束训练目标，把格律逐步内化进模型。

## 一、项目目标

纯 Python + numpy，**除 numpy 与标准库外零第三方运行时依赖**（GPU 加速与语料清洗所需的依赖均为可选），从 0 到 1 实现：

1. 一个**真能训练**的微型 GPT：字符级分词、手写前向 / 反向传播、手写 AdamW、KV Cache 增量解码
2. 一套**格律内化的训练与工具链**：
   - 平水韵韵书构建与逐位置元信息（韵律标签、损失权重）生成
   - 加权交叉熵主损失 + 平仄头 / 韵部头辅助损失
   - 跨里程碑续训（按名载参）、真续训 checkpoint、遗忘基线红线监控
   - BoN 自蒸馏混合再训
   - 格律评估与 A/B 对照、句长约束采样

形态为「训练 + 工具链」，不含 HTTP 推理服务与命令行生成器。

## 二、系统架构

```
┌──────────────── 数据与词表层（tools/）────────────────┐
│ build_pingshui.py → data/pingshui.json（106 韵部）    │
│ build_corpus.py   → data/corpus.txt / data/val.txt     │
│                     / data/meta.npz（逐位置元信息）    │
│ migrate_vocab.py  → model-ext.npz（扩词表按名继承）    │
└──────────────────────────┬────────────────────────────┘
                           ▼
┌──────────────── 训练层（train.py + model/）───────────┐
│ 抽题（含逐位置权重 / 平仄 / 韵部标签与掩码）           │
│   → GPT 前向 → 加权主损失 + 平仄/韵部辅助损失          │
│   → 手写反向传播 → 手写 AdamW → 真续训 checkpoint     │
│   → val loss / val_ratio 监控 + 定期生成目检           │
└──────────────────────────┬────────────────────────────┘
                           ▼
┌──────────────── 里程碑与工具层 ───────────────────────┐
│ M1：扩词表续训 + 挂载韵部头                            │
│ M2：在 M1 之上再挂载平仄头                             │
│ M3：tools/selftrain.py —— BoN 自蒸馏混合再训           │
│ 评估：tools/prosody_eval.py —— 旧档 / M1 / M2 / M3     │
└────────────────────────────────────────────────────────┘
```

采样期可用 `meter.py` 的句长约束处理器（作为 `logits_processor` 挂在 `GPT.generate` 上）强制半句字数与断句。

## 三、模型规模

分词为 char-level。词表 = 语料纯文本字符表（8196 项）+ 5 个诗体控制 token（`<五绝>`、`<七绝>`、`<五律>`、`<七律>`、`<杂言>`，追加在末尾，id 8196..8200），合计 8201。

| 超参数 | 默认回归档（`train.py` 默认） | 里程碑档（M1 / M2 / M3） |
| -------- | -------- | -------- |
| d\_model | 64 | 256 |
| n\_head | 4 | 8 |
| n\_layer | 2 | 6 |
| ctx\_len | 64 | 128 |
| 词表 | 8201 | 8201 |
| 训练步数 | 3000 | M1 2000 / M2 1000 / M3 1000 |

* 默认回归档：`python train.py` 以默认超参启动（CPU 可训，用于回归冒烟）。注意：仓库根目录存在 d256 的 `model-ext.npz` 时，默认 d64 骨干会在 L\_base 锚定处因参数形状不符而中止（`load_params_partial` 按名载参的形状断言），该冒烟路径需显式传入与默认骨干同构的 `l_base_path`，或在不含该锚档的干净检出中运行。
* 里程碑档：由 `train.train(...)` 显式传参启动（见第九节），骨干 d256/h8/L6/ctx128，参数量约 900 万。
* 里程碑档的骨干必须与 `model-ext.npz` 一致，否则按名载参会因参数形状不符而断言失败。

## 四、目录结构

```
Minimal-LLM/
├── README.md            # 本文档（设计 + 使用说明）
├── PLAN.md              # 早期分阶段实现计划（Task 6/7 的 HTTP 推理服务与 CLI 未实现，不在本仓库范围内）
├── prosody.py           # 格律工具：平水韵查表（字→平仄/韵部）+ 句长/押韵/平仄/对仗判定 + 综合打分
├── meter.py             # 采样期句长约束处理器（挂到 GPT.generate 的 logits_processor）
├── train.py             # 数据加载 / 编码 / 加权损失抽题 / AdamW / 学习率调度 / 续训 checkpoint / 训练循环
├── model/
│   ├── __init__.py
│   ├── backend.py       # 计算后端总闸：默认 numpy(CPU)；MINIMAL_GPU=1 且探活通过时切 cupy(GPU)
│   ├── layers.py        # 原语：Linear / LayerNorm / GELU（含 forward+backward）
│   ├── attention.py     # 多头注意力：QKV 投影 + 缩放点积 + causal mask（含反向 + KV Cache + GQA）
│   └── gpt.py           # Block 堆叠 + GPT 组装；加权 loss / 辅助头 / generate / 保存加载
├── tools/
│   ├── build_pingshui.py # 平水韵韵书构建 → data/pingshui.json（+ 审计/冲突记录）
│   ├── build_corpus.py   # 全唐诗 JSON → data/corpus.txt + data/val.txt + data/meta.npz
│   ├── migrate_vocab.py  # 旧档 model.npz(V0=8196) → model-ext.npz(V=8201)
│   ├── selftrain.py      # BoN 自蒸馏：prompt 池 → 候选生成 → 格律筛选 → 混合集 → 混合再训（M3）
│   └── prosody_eval.py   # 格律内化评估与 A/B 对照（旧档 / M1 / M2 / M3）
├── data/                # 语料、韵书、逐位置元信息、自蒸馏产物（详见下）
├── test/                # 单元测试（18 个用例文件，自带逐项输出与退出码约定：✓/✗ 或 PASS 风格）
├── logs/                # 训练运行日志（本地产物，不入库）
└── _probe/              # 构建期原始数据与离线依赖（本地产物，不入库）
```

`data/` 下的产物：

```
data/
├── pingshui.json          # 平水韵韵书（106 韵部），tools/build_pingshui.py 生成
├── pingshui_audit.txt     # 韵书构建审计（源支持数分布 / 丢弃清单）
├── pingshui_conflicts.txt # 韵书冲突字记录（多韵部 / 跨源不一致）
├── corpus.txt             # 训练语料（诗体标签 + 正文），tools/build_corpus.py 生成
├── val.txt                # 独立验证集（400 首，与训练集同构），同源生成
├── meta.npz               # 逐位置元信息四数组，与 corpus.txt 的 token 流等长
├── eval-prompts.txt       # 评估 prompt 池，tools/prosody_eval.py 抽取后落盘复用
├── selftrain-prompts.txt  # 自蒸馏 prompt 池（每行「诗体\t首半句」）
├── selftrain.txt          # 过滤后的自产集
├── selftrain-meta.npz     # 自产集逐位置元信息
├── mix.txt                # 原语料 + 自产集循环 R 次的混合训练集
├── mix-meta.npz           # 混合集逐位置元信息
└── _selftrain_raw.jsonl   # 生成阶段断点账本（每完成一条 prompt 追加一行）
```

`data/meta.npz` 四数组与语料 token 流逐位置对齐，dtype 与含义如下：

| 数组 | dtype | 含义（下标 = token 在语料中的绝对位置） |
| -------- | -------- | -------- |
| weights | float32 | 该位置作为预测目标时的主损失权重；取值 `{1.0, 1.5, 2.0}`，分别为常态、半句末标点、韵脚 |
| tone_labels | int8 | 平仄标签：0=平、1=仄、2=韵书未覆盖或非汉字 |
| rhyme_labels | int16 | 韵部 id：0=unknown、1..106=韵部 |
| zone_tags | int8 | 位置属性：0=常态、1=韵脚、2=半句末标点、3=特殊 token、4=对仗区 |

`zone_tags` 重叠时取优先级「特殊 token(3) > 对仗区(4) > 韵脚(1) > 半句末标点(2) > 常态(0)」。韵脚 = 偶半句（第 2/4/6/8 半句）末字；对仗区 = 仅 `<五律>` / `<七律>` 的第 2、3 行全部汉字。

## 五、落地概念一览（数据 → 训练 → 格律）

| 类别 | 概念 | 落地位置 |
| -------- | -------- | -------- |
| 数据 | 诗体控制 token（首行前缀） | tools/build_corpus.py `POEM_TAGS` / `tag_of` / `tag_poem` |
| 数据 | 繁→简、校注剥离、白名单净化 | tools/build_corpus.py `clean_para` |
| 数据 | 逐位置元信息四数组 | tools/build_corpus.py `build_meta` |
| 数据 | 平水韵韵书（字→平仄 / 字→韵部） | tools/build_pingshui.py → data/pingshui.json |
| 分词 | char-level 词表 + 5 个特殊 token | train.py `SPECIAL_TOKENS` / `load_corpus` / `encode` |
| 词表 | 扩词表迁移（扩行/扩列 + 新行/新列全零） | tools/migrate_vocab.py `migrate` |
| 训练 | 加权交叉熵主损失（logsumexp 数值稳定） | model/gpt.py `loss` |
| 训练 | 平仄头 / 韵部头辅助损失（按掩码加权） | model/gpt.py `aux_loss` / `_masked_ce` |
| 训练 | 辅助头逐位置掩码派生 | train.py `derive_masks` / `get_batch_ext` |
| 训练 | 手写反向传播（含辅助头梯度在 h 处相加） | model/{layers,attention,gpt}.py `backward` |
| 训练 | 手写 AdamW（偏差校正 + 解耦权重衰减） | train.py `AdamW` |
| 训练 | 学习率：线性热身 + 余弦退火（按全局步数） | train.py `get_lr` |
| 训练 | 真续训 checkpoint（参数 + m/v/t + 词表快照 + 调度签名） | train.py `save_checkpoint` / `load_checkpoint` |
| 训练 | 跨里程碑按名载参 | train.py `load_params_partial` |
| 训练 | 遗忘红线：val\_ratio = val\_loss / L\_base | train.py `train`（L\_base 锚定 model-ext.npz） |
| 格律 | 句长合规率 / 押韵 / 平仄 / 对仗判定 | prosody.py `check_line_len` / `check_rhyme` / `check_tone` / `check_duizhang` |
| 格律 | 候选综合打分（含模型 logprob 注入） | prosody.py `score_candidate` |
| 格律 | 采样期句长硬约束 | meter.py `make_meter_processor` |
| 自蒸馏 | prompt 池隔离 + 候选筛选 + 混合配比 | tools/selftrain.py |
| 评估 | 指标固化 + 置信区间 + 闸门 + A/B 报告 | tools/prosody_eval.py |

## 六、数据流

1. `python tools/build_pingshui.py`：多源韵表并集交叉校验 → `data/pingshui.json`（106 韵部），并落盘审计与冲突记录。
2. `python tools/build_corpus.py`：全唐诗 JSON 分卷 → 清洗 → 打诗体标签 → 随机划分 → `data/corpus.txt`、`data/val.txt`、`data/meta.npz`（语料与 meta 由同一次遍历产出，保证等长）。
3. `python tools/migrate_vocab.py`：旧档 `model.npz`（V0=8196）→ `model-ext.npz`（V=8201），供带特殊 token 的续训使用。
4. M1：以 `model-ext.npz` 为基线按名载入共有参数、挂载韵部头续训 → `model-gelv-m1.npz`。
5. M2：以 `model-gelv-m1.npz` 为基线，再挂载平仄头续训 → `model-gelv-m2.npz`。
6. M3：`python tools/selftrain.py` 以 `model-gelv-m2.npz` 为基线，用自产集与原语料按配比拼出 `data/mix.txt` + `data/mix-meta.npz` 做混合再训 → `model-gelv-m3.npz`。
7. `python tools/prosody_eval.py`：以 `data/eval-prompts.txt` 为 prompt 池，对旧档 / M1 / M2 / M3 用同一采样流逐条续写并计算格律指标，写盘 A/B 报告。

## 七、边界与错误处理

* 缺 numpy → 导入 `model.backend` 时即报错（项目运行时依赖仅 numpy + 标准库）。
* checkpoint 缺失 → `load_checkpoint` 抛 `FileNotFoundError`，消息含缺失路径；旧格式档（无 `_ckpt_version` 元数据）→ 抛 `ValueError` 拒载，要求从头训。
* 续训校验：档内词表与当前语料不一致、调度签名（warmup\_steps / peak\_lr 不等、anneal\_total 缩短）不一致 → 抛 `ValueError` 拒载，不静默错位。
* 跨里程碑续训**严禁**用 `load_checkpoint`，必须用 `load_params_partial`（按名只载双方共有的参数，新增头保留初始化）。原因：`load_checkpoint` 会逐名读取 `_opt.m.{参数名}`，M1 档缺 `_opt.m.tone_head.*` 等键会直接 `KeyError`，且其词表快照与调度签名校验也不适用于跨档续训。
* meta 与语料长度不一致 → `train.train` 内直接断言失败（不退化）；自定义语料须显式传入与之匹配的 `meta_path` 或重建 meta。
* `save_every` 必须等于 `print_every`，否则抛 `ValueError`；`n_rhyme > 0` 时必须等于 `rhyme_classes`，否则断言失败。
* 词表外字符（OOV）：`encode` / `load_val` 跳过，评估与训练口径一致。
* 韵书缺失 → `prosody` 抛 `FileNotFoundError`，提示先运行 `python tools/build_pingshui.py`。
* GPU 探活失败（cupy 可导入但驱动 / runtime 异常）→ 打印警告并自动回退 numpy（CPU），CPU 主路径零 GPU 交互。
* 默认 d64 骨干训练时若仓中存在 d256 的 `model-ext.npz`，L\_base 锚模型载参会因参数形状不符而断言失败；如需以默认骨干运行，请显式传入一个与骨干匹配（或不存在，从而跳过该项）的 `l_base_path`。

## 八、验证标准

1. 单元测试全部通过：每个测试脚本自带逐项输出（`✓/✗` 或 `PASS` 风格）与汇总，全部通过时进程退出码为 0，任一失败时以非零码退出。
   * 多数用例（如 `test_corpus.py`、`test_prosody.py`、`test_selftrain.py`）逐项打印 `✓ 函数名` 或 `✗ 函数名: 异常`，末尾输出中文汇总，任一失败即非零退出。
   * 早期用例（`test_layers.py`、`test_attention.py`、`test_gpt.py`、`test_kv_cache.py`、`test_meter.py`、`test_checkpoint.py`）为断言式，失败抛 `AssertionError` 中断（非零退出），通过时打印 `PASS` / 「梯度检查全部通过」等。
   * `test_backend_gpu.py` 在 GPU 不可用时打印「跳过（GPU 不可用）」并以 0 退出。
2. 生成文本有语言规律（非乱码），训练日志中的 `val loss` 随步数下降。
3. 遗忘红线：`val_ratio = val_loss / L_base`，其中 `L_base` 锚定扩词表迁移档 `model-ext.npz`（同 ctx / 同 val 口径）；连续 2 次 `val_ratio > 1.05` 时运行期学习率标量 `lr_scale` 折半。
4. 格律评估闸门（`tools/prosody_eval.py`）：M1 句长合规率 ≥ 95% 且押韵合规率 ≥ 90%；M2 平仄合规率 ≥ 85% 且较 M1 显著提升。比例指标样本硬下限为每诗体 ≥ 30、合计 ≥ 120；对仗指标须先过区分度自检：旧模型「联内不重字率」≥ 90%（近饱和）时该子指标作废、不参与有效性判定，`valid` 仅在参与判定的子指标中「至少一项差值 ≥ 10 个百分点」方可作为闸门依据。

## 九、使用方式

以下命令在项目根目录（`Minimal-LLM/`）下执行。`python` 指本机 Python 3 解释器。

### 1. 数据与词表准备（首次）

```bash
python tools/build_pingshui.py     # 生成 data/pingshui.json（需联网抓取三源韵表）
python tools/build_corpus.py       # 生成 data/corpus.txt、data/val.txt、data/meta.npz
python tools/migrate_vocab.py      # model.npz(V0=8196) -> model-ext.npz(V=8201)
```

* `build_corpus.py` 需先安装繁简转换依赖 `pip install opencc-python-reimplemented`，并把全唐诗 JSON 分卷（`poet.tang.*.json`）放入 `_probe/raw/`（该目录不入库）。
* `migrate_vocab.py` 需已存在旧档 `model.npz`。

### 2. 训练

```bash
# 默认回归档：train() 全默认超参（d64/h4/L2/ctx64，3000 步），CPU 可训
python train.py
```

* 注：仓库根目录存在 d256 的 `model-ext.npz` 时，上述默认 d64 冒烟会在 L\_base 锚定处因参数形状不符而中止；此时需显式传入与默认骨干同构的 `l_base_path`（详见第三节）。

里程碑训练由 `train.train(...)` 显式传参启动（未列出的超参沿用 `train.train` 形参默认值）：

```bash
# M1：扩词表续训 + 挂载韵部头（按名载入 model-ext.npz 的共有参数）
python -c "import train; train.train(base_path='model-ext.npz', ckpt_path='model-gelv-m1.npz', d_model=256, n_head=8, n_layer=6, ctx_len=128, n_rhyme=107, rhyme_classes=107, max_steps=2000, anneal_total=2000)"

# M2：在 M1 基础上再挂载平仄头
python -c "import train; train.train(base_path='model-gelv-m1.npz', ckpt_path='model-gelv-m2.npz', d_model=256, n_head=8, n_layer=6, ctx_len=128, n_rhyme=107, use_tone=True, rhyme_classes=107, max_steps=1000, anneal_total=1000)"
```

* `base_path` 为跨里程碑按名载参（`load_params_partial`），`ckpt_path` 为本次产出的 checkpoint；中途中断也会先落盘断点，续训时用 `resume_path` 指向该文件。
* 挂载辅助头：`n_rhyme` 控制韵部头类别数（> 0 时须等于 `rhyme_classes`），`use_tone=True` 挂载平仄头。
* 辅助损失权重 `lam_tone`（默认 0.3）、`lam_rhyme`（默认 0.5）；对仗区权重 `duizhang_weight`（取最大融合，M3 起设 1.5）。

### 3. BoN 自蒸馏（M3）

```bash
python tools/selftrain.py --stage all              # 生成 -> 构建混合集 -> 混合再训
python tools/selftrain.py --stage gen --n 500      # 仅生成候选（可中断续跑）
python tools/selftrain.py --stage build --n 500    # 仅构建自产集与混合集
python tools/selftrain.py --stage train            # 仅混合再训
```

可选参数：`--n`（prompt 总数，默认 500，按诗体均分）、`--k`（每 prompt 候选数，默认 8）、`--new-tokens`（每候选续写长度，默认 96）、`--model`（基线模型档，默认 `model-gelv-m2.npz`）。

### 4. 格律评估与 A/B 对照

```bash
python tools/prosody_eval.py                                    # 正式全量 A/B
python tools/prosody_eval.py --limit 2 --report _probe/ab_smoke.txt   # 冒烟（每诗体仅取前 2 条）
python tools/prosody_eval.py --models old,m2 --report test/output-gelv-ab.txt
```

可选参数：`--models`（逗号分隔，取值 `old,m1,m2,m3,m3_cons`，默认按模型档存在性自动选择）、`--limit`（每诗体取前 N 条）、`--n-per-tag`（每诗体 prompt 条数，默认 40）、`--new-tokens`（每首续写 token 数，默认 96）、`--report`（报告落盘路径，默认 `test/output-gelv-ab.txt`）。

### 5. GPU 加速（可选）

默认纯 numpy/CPU；设置环境变量 `MINIMAL_GPU=1` 且已安装 cupy、运行期探活通过时，全链路切到 GPU：

```powershell
$env:MINIMAL_GPU=1; python train.py
$env:MINIMAL_GPU=1; python tools/selftrain.py --stage train
$env:MINIMAL_GPU=1; python tools/prosody_eval.py
```

探活失败（驱动 / runtime 异常）会自动回退 CPU 并打印警告。A/B 各组必须运行在同一设备上，否则 `prosody_eval.py` 会报错。

### 6. 单元测试（逐个子测 + 汇总）

```bash
python test/test_layers.py
python test/test_attention.py
python test/test_gpt.py
python test/test_kv_cache.py
python test/test_meter.py
python test/test_checkpoint.py
python test/test_weighted_loss.py
python test/test_aux_heads.py
python test/test_batch_ext.py
python test/test_tokenizer.py
python test/test_corpus.py
python test/test_pingshui.py
python test/test_prosody.py
python test/test_critic.py
python test/test_vocab_migrate.py
python test/test_selftrain.py
python test/test_prosody_eval.py
python test/test_backend_gpu.py
```

汇总运行（PowerShell，逐个运行并汇报失败项）：

```powershell
Get-ChildItem test -Filter "test_*.py" | ForEach-Object {
    python "test/$($_.Name)"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($_.Name)（退出码 $LASTEXITCODE）" }
}
```
