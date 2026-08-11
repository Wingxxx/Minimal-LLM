# 极简 LLM：从 0 到 1 搭建微型 GPT 并用推理服务包装

> 作者：WING · 2026-08-11
> 定位：把 ai-learning 学过的 Transformer 前向、KV Cache、连续批处理、SSE 流式、p99 优化
> 全部落地成一个能跑通「训练 → 推理 → 服务化」的极简系统。

## 一、项目目标

纯 Python + numpy，**零第三方依赖**，从 0 到 1 实现：

1. 一个**真能训练**的微型 GPT（char-level，手写 forward/backward）
2. 一个**服务化包装**：HTTP 推理服务（SSE 流式 + 动态批处理 + KV Cache + p99 统计）+ CLI

## 二、系统架构

```
┌───────────────────────── 训练侧 train.py ─────────────────────────┐
│  data/corpus.txt → 字符分词 → GPT 前向 → 交叉熵 loss               │
│                      → 手写反向传播 → 手写 AdamW → 存 model.npz    │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼ 加载权重
┌───────────────────────── 推理侧 server.py ────────────────────────┐
│  HTTP 服务 (/v1/completions)                                      │
│    → 请求队列 → 连续批处理调度器（满批/超时触发发批）                │
│    → 组批前向（KV Cache：decode 阶段只算新 token）                  │
│    → SSE 逐 token 推送（text/event-stream） 或 非流式 JSON          │
│    → 采样（temperature / top-k / top-p）→ p99 延迟日志              │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼
┌───────────────────────── cli.py ─────────────────────────────────┐
│  命令行交互：python cli.py "prompt" → 本地加载权重直接生成           │
└───────────────────────────────────────────────────────────────────┘
```

## 三、模型规模（极简但能学会语言）

| 超参数      | 取值         | 说明                    |
| -------- | ---------- | --------------------- |
| 分词       | char-level | 词表 = 语料中出现过的字符集合（约 2600 个） |
| d\_model | 64         | 嵌入维度                  |
| n\_head  | 4          | 注意力头数，head\_dim = 16  |
| n\_layer | 2          | Transformer 层数        |
| 上下文      | 64         | 单次看到的 token 数         |
| 参数量      | \~27 万     | CPU 可训                |
| 训练步数     | 3000 步        | 《唐诗三百首》正文（约 2.7 万字符），CPU 5-10 分钟 |

## 四、目录结构

```
Minimal-LLM/
├── README.md            # 本文档（设计 + 使用说明）
├── data/corpus.txt      # 训练语料（《唐诗三百首》319 首，约 2.7 万字符）
├── model/
│   ├── __init__.py
│   ├── layers.py        # 原语：Linear / LayerNorm / GELU（含 forward+backward）
│   ├── attention.py     # 多头注意力：QKV 投影 + 缩放点积 + causal mask（含反向 + KV Cache + GQA）
│   └── gpt.py           # Block（Attn + FFN + 残差 + LN）+ GPT 组装；loss/generate/保存加载
├── train.py             # 训练循环：手写 AdamW、loss 打印、保存 model.npz
├── server.py            # HTTP 推理服务：SSE 流式 + 动态批处理调度器 + KV Cache + p99
├── cli.py               # 命令行交互
└── test/
    ├── test_layers.py       # 原语梯度检查（有限差分）
    ├── test_attention.py    # MHA / KV Cache / GQA 测试
    ├── test_gpt.py          # loss / 生成 / 保存加载 测试
    ├── test_train.py        # 训练冒烟测试
    ├── test_server.py       # 服务集成测试（流式/并发/p99）
    ├── output-train.txt     # 训练日志（独立 output 文件）
    ├── output-cli.txt       # CLI 生成演示输出
    └── output-server.txt    # 服务端请求/延迟日志
```

## 五、落地概念一览（训练 → 模型 → 推理服务）

| 类别 | 概念 | 落地位置 |
| -------- | -------- | -------- |
| 训练 | 字符级分词（char-level 词表） | train.py `load_corpus()` |
| 训练 | 交叉熵损失 | model/gpt.py `loss()` |
| 训练 | 手写反向传播（解析梯度） | model/{layers,attention,gpt}.py `backward()` |
| 训练 | 手写 AdamW（偏置修正 + 权重衰减） | train.py `AdamW` |
| 模型 | 多头注意力（缩放点积 + causal mask + GQA 分组 KV） | model/attention.py |
| 模型 | LayerNorm / GELU / 残差连接 | model/layers.py、model/gpt.py `Block` |
| 模型 | 可学习位置嵌入 | model/gpt.py `pos_emb` |
| 模型 | 采样：temperature / top-k / top-p | model/gpt.py `_sample()` |
| 模型 | 模型保存 / 加载（model.npz） | model/gpt.py `save` / `load` |
| 推理服务 | KV Cache（decode 阶段只计算新 token） | model/attention.py |
| 推理服务 | 连续批处理调度器（满批 / 超时触发） | server.py `Batcher` |
| 推理服务 | SSE 流式逐 token 推送 | server.py |
| 推理服务 | p99 延迟统计（avg / p50 / p99） | server.py `Tracker` |

> 延伸练习（不入主线）：正弦位置编码 / MoE / SFT 微调 / 混合精度 / int8 量化，见 PLAN.md 附录①-⑤，均为独立文件、独立验证。

## 六、数据流

1. `python train.py`：语料 → 训练 → 保存 `model.npz`，输出日志到 test/output-train.txt
2. `python server.py`：加载权重 → 监听 HTTP → 日志到 test/output-server.txt
3. `curl` / 浏览器请求 → 调度器组批 → KV Cache 前向 → SSE 逐 token → p99 日志
4. `python cli.py "prompt"`：本地加载权重直接生成，输出到 test/output-cli.txt

## 七、边界与错误处理

* 无 numpy 或 checkpoint 缺失 → 明确报错并提示先运行 `python train.py`

* 上下文超长 → 截断到模型最大长度

* 服务端参数校验：max\_tokens / temperature 范围检查

* 模型极小，无 OOM 风险

## 八、验证标准

1. 训练后生成文本**有语言规律**（非乱码，能看出词与词的关联）
2. 服务端流式 / 非流式请求均能正确返回
3. 每次运行都落独立 output 文件（代码必有 output）
4. 并发请求下调度器正确组批，p99 日志可读

## 九、使用方式（实现后生效）

```bash
cd Minimal-LLM
python train.py            # 训练（首次必跑）
python cli.py "床前明月光"   # 命令行生成
python server.py           # 起服务后 curl http://127.0.0.1:8000/v1/completions
```

