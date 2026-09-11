"""句长约束（格律）处理器：为自回归采样提供「五言/七言」硬约束。

只依赖计算后端 np，不感知模型结构；作为 logits_processor 回调挂在 GPT.generate
的采样前一步，对得分做掩码（置 -inf 的 token 永不被抽中）。

规则（三态，L = 目标半句汉字数，cur = 末尾半句已生成汉字数）：
  - cur <  L：屏蔽全部非汉字 token（换行与各类标点）→ 半句内只允许汉字；
              例外：上一 token 为「。」时额外放行一次换行符（每联换行，不计入句长）
  - cur == L：只留目标标点、其余全屏蔽 → 强制在此断句
  - cur >  L：不约束（提示词本身越界的兜底）
目标标点按半句奇偶：奇数半句（已出现标点数为偶）取「，」，偶数半句取「。」。

参数约定（本模块保持纯逻辑，字符语义由调用方给定）：
  - punct_ids     可作断句的标点 id（通常「，」「。」）
  - non_hanzi_ids 词表中全部非汉字 token id：既作计数断句符集合（不计入句长），
                  又在 cur < L 时被整体屏蔽；None 时退化为旧口径（仅屏蔽 punct_ids）。
                  前置条件：给定时应为词表**全部**非汉字（务必含 punct_ids 与 newline_id），
                  否则标点会被误计为汉字导致约束静默失效
  - newline_id    换行符 id，「。」断句后的下一步放行一次；None 时完全禁绝。
                  前置条件：给定时应包含在 non_hanzi_ids 内，否则换行仍会偷占句长额度
"""
from model.backend import np

DEFAULT_LINE_LEN = 5                      # 推断不出时的兜底句长（五言）
NEG_INF = float("-inf")                   # 掩码值：softmax 后概率恒为 0


def _count_trailing(ids, stop_set):
    """末尾半句已生成的汉字数：从尾部往前数，遇任一非汉字 token（stop_set）即止。"""
    n = 0
    for t in reversed(ids):
        if int(t) in stop_set:
            break
        n += 1
    return n


def _count_punct(ids, punct_set):
    """整段已出现的标点数（决定当前半句的奇偶序号）。"""
    return sum(1 for t in ids if int(t) in punct_set)


def infer_line_len(ids, stop_ids, default=DEFAULT_LINE_LEN):
    """从已给序列推断目标句长（半句汉字数）。

    优先级：① 末尾未断半句的汉字数（>0 直接用，五言/七言一目了然）
            ② 末尾为非汉字（标点/换行）→ 剥掉尾部全部非汉字，回退最后一个完整半句
            ③ 都取不到（空 / 全是非汉字）→ 返回 default
    """
    stop_set = set(int(p) for p in stop_ids)
    tail = _count_trailing(ids, stop_set)
    if tail > 0:
        return tail
    seq = list(ids)
    while seq and int(seq[-1]) in stop_set:       # 剥掉尾部非汉字后回退一个完整半句
        seq.pop()
    tail = _count_trailing(seq, stop_set)
    return tail if tail > 0 else default


def make_meter_processor(punct_ids, comma_id, period_id, line_len=None,
                         non_hanzi_ids=None, newline_id=None):
    """构造采样期 logits 处理器，签名 (logits[V], idx[1,T]) -> logits[V]。

    line_len=None 时在**首次调用**用 infer_line_len(idx[0], stop_ids) 推断并冻结
    （句长必须在生成全程恒定，逐次重推会被新生成的半句带偏）。
    line_len=0 或 punct_ids 为空 → 返回 None（调用方视同无约束）。
    """
    if not punct_ids:
        return None
    punct_list = [int(p) for p in punct_ids]
    punct_set = set(punct_list)
    comma_id, period_id = int(comma_id), int(period_id)
    if non_hanzi_ids is None:                    # 退化口径：仅标点算断句符、仅屏蔽标点
        stop_list = punct_list
    else:
        stop_list = [int(x) for x in non_hanzi_ids]
    stop_set = set(stop_list)
    newline_id = None if newline_id is None else int(newline_id)
    state = {"L": None if line_len is None else int(line_len)}
    if state["L"] == 0:
        return None

    def processor(logits, idx):
        if state["L"] is None:                   # 首次调用：推断后冻结
            state["L"] = infer_line_len(idx[0], stop_list)
        L = state["L"]
        ids = idx[0]
        cur = _count_trailing(ids, stop_set)
        if cur > L:
            return logits                        # 提示词已越界：兜底放行
        out = logits.copy()
        if cur < L:                              # 禁止提前断句：屏蔽全部非汉字
            for i in stop_list:
                out[i] = NEG_INF
            # 「。」断句后的下一步放行一次换行（每联换行；换行不计入句长）
            if newline_id is not None and len(ids) > 0 and int(ids[-1]) == period_id:
                out[newline_id] = logits[newline_id]
            return out
        # cur == L：强制断句——除目标标点外全部屏蔽
        target = comma_id if (_count_punct(ids, punct_set) % 2 == 0) else period_id
        out[:] = NEG_INF
        out[target] = logits[target]
        return out

    return processor
