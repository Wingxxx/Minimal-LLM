# -*- coding: utf-8 -*-
"""平水韵韵书构建管线：多源并集交叉校验 → data/pingshui.json（106 韵部）

用途：为「平仄 / 押韵」训练期内化提供字→韵部→声调映射，替代推理期硬掩码。
本脚本仅依赖标准库（urllib 抓取 + json/re/collections），不引入任何第三方运行时依赖；
繁体→简体字形归一仅在构建期读取 opencc 字典 TSCharacters.txt 的文本，不 import opencc。

────────────────────────────────────────────────────────────────────────
一、数据来源（3 个独立公开源，逐字多源交叉；实测三源体量悬殊）
  1) dockerian   https://dockerian.github.io/poetry/gelv/html/pingshui.html
     —— HTML 韵表，106 组；实测去重 14336 字，主键以简体为主（如「東」出现 0 次）
  2) charlesix59 https://raw.githubusercontent.com/charlesix59/chinese_word_rhyme/main/data/Pingshui_Rhyme.json
     —— JSON 韵表，105 组（缺「上声三讲」）；实测去重 8079 字，主键亦以简体为主
  3) weihethu    https://raw.githubusercontent.com/weihethu/chinese-poem-composer/master/pingshuiyun.txt
     —— 简体韵表，106 组（版式含续行/异名，按序号单调规则归位）；实测去重 4349 字
  抓取结果缓存于 _probe/pingshui_raw/（已 gitignore）；缓存存在则跳过下载。
  直连失败时回退本机代理 127.0.0.1:7897。结果一次性落盘后不再联网。

二、交叉规则（多源并集，替代原「≥2 源一致」口径）
  实测三源体量失衡，原「某字在韵部被 ≥2 源同时列出方采用」会误杀单源合法字
  （如「啼」繁简同形、仅 dockerian 收录），故修正为：
    -「某源未收录」≠「各源矛盾」：仅当同字在不同源被分到不同韵部，才算跨源冲突；
    - 采用规则：任一源收录即纳入；其韵部取各源并集（多值显式保留在 JSON 各韵部下）；
    - 跨源韵部不一致之字，逐字记入 data/pingshui_conflicts.txt（含各源原始归属）；
    - 置信度留痕：data/pingshui_audit.txt 记录每字「源支持数」分布与被丢弃字清单，
      杜绝静默丢弃；
    - 兜底：非基本汉字区（U+4E00–U+9FFF 以外）字符丢弃并记入审计；任一韵部为空即 raise。

三、字形归一（可选加固）
  t2s 降级为可选：构建期读取 TSCharacters.txt 文本（不 import opencc）。
    - 仅当候选唯一时才归一；多候选保留原字形为键并记入 conflicts
      （禁止「多候选取首项」，否则「乾 → 干」会把 乾 的韵部错投给 干，造成语义污染）；
    - 唯一候选若落在基本汉字区外（opencc 对生僻繁体常映射到扩展区），回退保留原字形，
      避免归一反而误杀合法字；
    - 字典文件缺失 → 恒等映射 + stderr 告警 + 继续构建，不硬依赖已 gitignore 的 _probe/。

四、多读字策略（不静默取其一）
  同一字若对应多个不同韵部（多音/多读），不丢弃、不静默取其一：
    - JSON 中该字同时保留在所属的每个韵部下（显式多值）；
    - 逐字记入 conflicts（字名 + 冲突类型 + 各源原始归属）。
  评估口径为「任一读音匹配即合规」——下游做平仄/押韵判定时，该字命中任一韵部即视为可查。

五、JSON Schema
  {
    "<韵部名>": {"tone": "平|上|去|入", "chars": ["东", "同", ...]},
    ...
  }
  韵部名 = 声部前缀 + 序号韵目，如「上平一东 / 下平十一尤 / 上声二肿 / 去声一送 / 入声六月」。
  声部与声调对应：上平/下平→平，上声→上，去声→去，入声→入。
  韵部总数恒为 106：平 30（上平 15 + 下平 15）、上 29、去 30、入 17。
  （仄 = 上/去/入 的合称，本数据不合并为「仄」，四声各自独立保留。）

六、覆盖率口径（token 级）
  分母 = corpus.txt 中所有 CJK 汉字（U+4E00–U+9FFF）出现总次数（标点/空白/非汉字不计）；
  分子 = 能在韵书查到（字出现在任一韵部）的汉字出现总次数。
  另报「去重字符覆盖率」与「未覆盖字 Top50 及频次」，供人工目检。
  脚本内建门禁：token 级覆盖率 < 95% 时以非零退出码中止（BLOCKED），不静默降级。
"""
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（脚本在 tools/ 下）
RAW_DIR = os.path.join(BASE, "_probe", "pingshui_raw")
TS_PATH = os.path.join(BASE, "_probe", "vendor", "pkg", "opencc", "dictionary", "TSCharacters.txt")
OUT_JSON = os.path.join(BASE, "data", "pingshui.json")
OUT_CONFLICT = os.path.join(BASE, "data", "pingshui_conflicts.txt")
OUT_AUDIT = os.path.join(BASE, "data", "pingshui_audit.txt")
CORPUS = os.path.join(BASE, "data", "corpus.txt")

# 文件名 -> 下载地址（缓存缺失时按此抓取）
SOURCES = {
    "dockerian_pingshui.html":
        "https://dockerian.github.io/poetry/gelv/html/pingshui.html",
    "charlesix59_Pingshui_Rhyme.json":
        "https://raw.githubusercontent.com/charlesix59/chinese_word_rhyme/main/data/Pingshui_Rhyme.json",
    "weihethu_pingshuiyun.txt":
        "https://raw.githubusercontent.com/weihethu/chinese-poem-composer/master/pingshuiyun.txt",
}
PROXIES = [None, "http://127.0.0.1:7897"]   # 直连优先，失败回退本机代理
MIN_COVERAGE = 0.95                         # token 级覆盖率硬门禁

CJK = re.compile(r"[\u4e00-\u9fff]")        # 基本汉字区（U+4E00–U+9FFF）

# 平水韵 106 韵目（按声部列出，顺序即序号顺序）
SECTIONS = {
    "上平": ["一东", "二冬", "三江", "四支", "五微", "六鱼", "七虞", "八齐", "九佳", "十灰",
             "十一真", "十二文", "十三元", "十四寒", "十五删"],
    "下平": ["一先", "二萧", "三肴", "四豪", "五歌", "六麻", "七阳", "八庚", "九青", "十蒸",
             "十一尤", "十二侵", "十三覃", "十四盐", "十五咸"],
    "上声": ["一董", "二肿", "三讲", "四纸", "五尾", "六语", "七麌", "八荠", "九蟹", "十贿",
             "十一轸", "十二吻", "十三阮", "十四旱", "十五潸", "十六铣", "十七筱", "十八巧",
             "十九皓", "二十哿", "二十一马", "二十二养", "二十三梗", "二十四迥", "二十五有",
             "二十六寝", "二十七感", "二十八琰", "二十九豏"],
    "去声": ["一送", "二宋", "三绛", "四寘", "五未", "六御", "七遇", "八霁", "九泰", "十卦",
             "十一队", "十二震", "十三问", "十四愿", "十五翰", "十六谏", "十七霰", "十八啸",
             "十九效", "二十号", "二十一个", "二十二祃", "二十三漾", "二十四敬", "二十五径",
             "二十六宥", "二十七沁", "二十八勘", "二十九艳", "三十陷"],
    "入声": ["一屋", "二沃", "三觉", "四质", "五物", "六月", "七曷", "八黠", "九屑", "十药",
             "十一陌", "十二锡", "十三职", "十四缉", "十五合", "十六叶", "十七洽"],
}
TONE_OF = {"上平": "平", "下平": "平", "上声": "上", "去声": "去", "入声": "入"}
# 规范韵部顺序：[("上平一东", "平"), ...]（用于稳定输出顺序）
CANON = [(sec + name, TONE_OF[sec]) for sec, names in SECTIONS.items() for name in names]
CANON_KEYS = [k for k, _ in CANON]


def read_text(path):
    """按编码回退读取文本（源站编码不一，weihethu 非 UTF-8）。"""
    for enc in ("utf-8", "gb18030", "big5"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("无法解码: " + path)


def ensure_sources():
    """确保原始数据存在；缺失则用标准库 urllib 抓取（直连 → 本机代理）并缓存。"""
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        for proxy in PROXIES:
            handler = {"http": proxy, "https": proxy} if proxy else {}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(handler))
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with opener.open(req, timeout=30) as r:
                    data = r.read()
                with open(path, "wb") as f:
                    f.write(data)
                print(f"[抓取] {name} <- {'直连' if proxy is None else proxy} ({len(data)} bytes)")
                break
            except Exception as e:  # noqa: BLE001 网络异常统一上报后换通道
                print(f"[重试] {name} via {'直连' if proxy is None else proxy}: "
                      f"{type(e).__name__}: {e}")
                time.sleep(1)
        else:
            raise RuntimeError(f"源抓取失败（全部通道）：{name} <- {url}")


def load_t2s():
    """读取 TSCharacters.txt 文本，返回 (唯一候选映射, 多候选字集合)。

    仅当候选唯一时才纳入映射；多候选字不归一（由调用侧保留原字形并记入 conflicts）。
    字典缺失时返回空映射 + 空集合，并打印 stderr 告警（优雅降级，不硬依赖 _probe/）。
    """
    if not os.path.exists(TS_PATH):
        print("[告警] 未找到 t2s 字典，退化为恒等映射（不做字形归一）：" + TS_PATH,
              file=sys.stderr)
        return {}, set()
    unique, multi = {}, set()
    with open(TS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            trad, simps = parts
            cands = simps.split()
            if len(cands) == 1:
                unique[trad] = cands[0]
            elif len(cands) > 1:
                multi.add(trad)
    return unique, multi


def _is_cjk_any(ch):
    """判断字符是否属 CJK 汉字各扩展区（用于区分「非 CJK 字符」与「非基本汉字区」）。"""
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2EBEF or 0x30000 <= o <= 0x323AF)


def parse_dockerian():
    """解析 dockerian HTML 韵表，返回 {规范键: [原始字符]}（仅去空白，未做 CJK 过滤）。

    注意：大韵部会被拆成多个 <p> 块，须取「h4 到下一个 h4 之间」的全部内容；
    「二十八俭」为「二十八琰」异名，归一为规范名。该源主键以简体为主。
    """
    html = read_text(os.path.join(RAW_DIR, "dockerian_pingshui.html"))
    sec_map = {"平声上": "上平", "平声下": "下平", "上声部": "上声",
               "去声部": "去声", "入声部": "入声"}
    out = {}
    parts = re.split(r'<h3 class="yun"[^>]*><span>([^<]+)</span></h3>', html)
    for i in range(1, len(parts), 2):
        sec = sec_map.get(parts[i])
        if sec is None:
            continue
        body = parts[i + 1]
        chunks = re.split(r'<h4 class="yun"[^>]*><span>([^<]+)</span></h4>', body)
        for j in range(1, len(chunks), 2):
            name, content = chunks[j], chunks[j + 1]
            if name == "二十八俭":        # 韵目异名，等同二十八琰
                name = "二十八琰"
            content = re.sub(r'<span class="note">.*?</span>', "", content, flags=re.S)
            content = re.sub(r'<[^>]+>', "", content)     # 剥掉其余标签
            out[sec + name] = [c for c in content if not c.isspace()]
    return out


def parse_charlesix():
    """解析 charlesix59 JSON 韵表，返回 {规范键: [原始字符]}（仅去空白，未做 CJK 过滤）。

    该源主键以简体为主。
    """
    with open(os.path.join(RAW_DIR, "charlesix59_Pingshui_Rhyme.json"), encoding="utf-8") as f:
        data = json.load(f)
    sec_map = {"上平声部": "上平", "下平声部": "下平", "上声部": "上声",
               "去声部": "去声", "入声部": "入声"}
    out = {}
    for sec, groups in data.items():
        s = sec_map.get(sec)
        if s is None:
            raise ValueError("charlesix59 出现未知声部：" + sec)
        for name, chars in groups.items():
            nm = name[2:] if name.startswith("入声") else name   # 剥离「入声」前缀（入声部内冗余）
            out[s + nm] = [c for c in chars if not c.isspace()]
    return out


def parse_weihethu():
    """解析 weihethu 简体韵表，返回 {规范键: [原始字符]}（仅去空白，未做 CJK 过滤）。

    该源版式有三类瑕疵，统一按「序号单调」规则归位：
      - 序号与上一组相同 → 视为续行并入上一组（如「四缁」续「四支」）；
      - 序号重置为「一」→ 新声部起点；
      - 组名可能为异名或无韵目字，故一律以「序号 + 当前声部」定位规范名。
    """
    txt = read_text(os.path.join(RAW_DIR, "weihethu_pingshuiyun.txt"))
    txt = txt.replace("\ufeff", "").replace("　", "")
    segs = txt.split("，")
    name_re = re.compile(r'([一二三四五六七八九十]+)([\u4e00-\u9fff]?)$')
    heads = [segs[0].strip()]
    bodies = []
    for s in segs[1:-1]:
        m = name_re.search(s)
        if not m:
            raise ValueError("weihethu 组头解析失败: " + repr(s[-20:]))
        heads.append(m.group(0))
        bodies.append(s[:m.start()])
    bodies.append(segs[-1])            # 末组正文（其后无组头）

    num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
           "九": 9, "十": 10}

    def ord_of(h):
        s = re.match(r'([一二三四五六七八九十]+)', h).group(1)
        if s == "十":
            return 10
        if "十" in s:
            a, b = s.split("十")
            return (num[a] if a else 1) * 10 + (num[b] if b else 0)
        return num[s]

    secs = list(SECTIONS)
    out, sec_idx, prev = {}, 0, 0
    for h, body in zip(heads, bodies):
        o = ord_of(h)
        if o == 1 and out:                     # 序号重置 → 进入下一声部
            sec_idx += 1
        if o == prev and out:                  # 序号重复 → 续行并入上一组
            key = next(reversed(out))
            out[key] += body
        else:
            out[secs[sec_idx] + SECTIONS[secs[sec_idx]][o - 1]] = body
        prev = o
    return {k: [c for c in v if not c.isspace()] for k, v in out.items()}


def normalize_sources(parsers, t2s):
    """对各源字表做字形归一，并把被丢弃字符按原因分类。

    归一规则：命中 t2s 且候选唯一时才替换为简体；多候选保留原字形。
    仅基本汉字区（U+4E00–U+9FFF）字形进入字表，其余记入 extra（带丢弃原因）。
    返回 (simp, extra, multi_hit)：
      simp     {源名: {韵部: set(基本汉字区字)}}；
      extra    {原因: Counter(字符 -> 出现次数)}；
      multi_hit 命中「t2s 多候选」而保留原字形之字集合。
    """
    unique, multi = t2s
    simp, extra, multi_hit = {}, defaultdict(Counter), set()
    for tag, src in parsers.items():
        d = {}
        for key, chars in src.items():
            keep = set()
            for c in chars:
                if c in multi:                       # 多候选：保留原字形，不归一
                    multi_hit.add(c)
                    if CJK.match(c):
                        keep.add(c)
                    else:
                        extra["t2s 多候选且非基本汉字区"][c] += 1
                elif CJK.match(c):
                    cand = unique.get(c, c)
                    # 候选若落在基本汉字区外（opencc 对生僻繁体常映射到扩展区），
                    # 归一会把合法字挤出可表示范围，故回退保留原字形，避免误杀。
                    keep.add(cand if CJK.match(cand) else c)
                elif _is_cjk_any(c):
                    extra["非基本汉字区（CJK 扩展）"][c] += 1
                else:
                    extra["非 CJK 字符"][c] += 1
            d[key] = keep
        simp[tag] = d
    return simp, extra, multi_hit


def cross_validate(simp):
    """多源并集交叉：返回 (accepted, char_groups, src_groups, support)。

    accepted:    {韵部: set(字)} —— 各源并集（任一源收录即纳入）；
    char_groups: {字: set(韵部)} —— 该字所属全部韵部（并集）；
    src_groups:  {字: {源名: set(韵部)}} —— 各源原始归属，用于冲突与归属留痕；
    support:     {字: 源支持数} —— 收录该字的源个数（同一源多韵部只计 1）。
    任一韵部并集后为空即 raise（不静默降级）。
    """
    char_groups, src_groups, support = defaultdict(set), defaultdict(dict), Counter()
    for tag in simp:
        seen = set()
        for key in CANON_KEYS:
            for ch in simp[tag].get(key, ()):
                char_groups[ch].add(key)
                src_groups[ch].setdefault(tag, set()).add(key)
                seen.add(ch)
        for ch in seen:
            support[ch] += 1

    accepted = {}
    for key in CANON_KEYS:
        acc = set()
        for tag in simp:
            acc |= simp[tag].get(key, set())
        if not acc:
            raise RuntimeError(f"韵部「{key}」多源并集后为空，拒绝静默降级")
        accepted[key] = acc
    return accepted, char_groups, src_groups, support


def classify_conflicts(char_groups, src_groups, multi_hit):
    """判定冲突字及类型，返回 {字: set(冲突类型)}。

    类型：多韵部（并集 >1 韵部）/ 跨源韵部不一致（不同源被分到不同韵部）/
    t2s 字形多候选保原字形。
    """
    reasons = defaultdict(set)
    for ch, gs in char_groups.items():
        if len(gs) > 1:
            reasons[ch].add("多韵部")
        sigs = {tuple(sorted(v)) for v in src_groups[ch].values()}
        if len(sigs) > 1:
            reasons[ch].add("跨源韵部不一致")
    for ch in multi_hit:
        if ch in char_groups:
            reasons[ch].add("t2s 字形多候选保原字形")
    return reasons


def build_records(accepted):
    """按规范顺序组装 JSON 记录（chars 去重排序）。"""
    return {key: {"tone": tone, "chars": sorted(accepted[key])} for key, tone in CANON}


def write_conflicts(reasons, src_groups):
    """冲突字逐字落盘（字 + 冲突类型 + 各源原始归属），返回条目数。"""
    def attribution(ch):
        return ";".join(f"{tag}:{','.join(sorted(gs))}"
                        for tag, gs in sorted(src_groups[ch].items()))

    items = sorted((ch, sorted(rs)) for ch, rs in reasons.items())
    with open(OUT_CONFLICT, "w", encoding="utf-8") as f:
        f.write("# 平水韵冲突字记录（多韵部 / 跨源韵部不一致 / t2s 字形多候选）\n")
        f.write("# 生成脚本：tools/build_pingshui.py；评估口径：任一读音匹配即合规\n")
        f.write("# 格式：字<TAB>冲突类型(逗号分隔)<TAB>各源原始归属(源:韵部1,韵部2;...)\n")
        f.write(f"# 条目数：{len(items)}\n")
        for ch, rs in items:
            f.write(f"{ch}\t{','.join(rs)}\t{attribution(ch)}\n")
    return len(items)


def _coverage(char_groups):
    """计算 token 级 / 去重字符覆盖率与未覆盖字列表（含频次）。"""
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    freq = Counter(c for c in text if CJK.match(c))
    known = set(char_groups)
    denom = sum(freq.values())
    hit = sum(n for ch, n in freq.items() if ch in known)
    dup = len(freq)
    dup_hit = sum(1 for ch in freq if ch in known)
    miss = [(ch, n) for ch, n in freq.most_common() if ch not in known]
    return {"denom": denom, "hit": hit, "dup": dup, "dup_hit": dup_hit,
            "miss": miss, "token": hit / denom if denom else 0.0}


def report_coverage(char_groups):
    """打印覆盖率报告，返回 token 级覆盖率（分母=语料 CJK 汉字出现总次数）。"""
    s = _coverage(char_groups)
    dup_cov = s["dup_hit"] / s["dup"] if s["dup"] else 0.0
    print(f"[覆盖率] token 级 {s['token']:.4%}（{s['hit']}/{s['denom']}） | "
          f"去重字符 {dup_cov:.4%}（{s['dup_hit']}/{s['dup']}）")
    print("[未覆盖 Top50] " + " ".join(f"{ch}{n}" for ch, n in s["miss"][:50]))
    return s["token"]


def write_audit(support, extra, miss, reasons):
    """落盘审计文件：源支持数分布 + 被丢弃字清单（含原因）+ 丢弃命中语料 token Top20。

    「被丢弃字」= 未能进入最终韵书的字符，含两类来源：
      (a) 源流中滤除的非基本汉字区字符（原因：非 CJK 字符 / 非基本汉字区（CJK 扩展））；
      (b) 语料中出现但三源均未收录的汉字（原因：源未收录）。
    """
    dist = Counter(support.values())
    n_multi = sum(1 for rs in reasons.values() if "多韵部" in rs)
    n_cross = sum(1 for rs in reasons.values() if "跨源韵部不一致" in rs)
    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write("# 平水韵构建审计（杜绝静默丢弃）\n")
        f.write("# 生成脚本：tools/build_pingshui.py\n#\n")
        f.write("# 【① 源支持数分布】\n")
        f.write("#   口径：某字在任一源任一韵部出现，即计该源 1 票；按「字」去重统计。\n")
        for n in (1, 2, 3):
            f.write(f"#   {n} 源支持：{dist.get(n, 0)} 字\n")
        f.write(f"#   采用字数合计：{sum(dist.values())} 字\n")
        f.write(f"#   其中多韵部字：{n_multi} 字；跨源韵部不一致字：{n_cross} 字\n#\n")

        f.write("# 【② 丢弃清单】（未能进入最终韵书的字符，逐类列明丢弃原因与出现次数）\n")
        total_drop = 0
        for reason in sorted(extra):
            cnt = extra[reason]
            total_drop += len(cnt)
            detail = " ".join(f"{c}({n})" for c, n in cnt.most_common())
            f.write(f"#   原因「{reason}」：{len(cnt)} 个，合计出现 {sum(cnt.values())} 次\n")
            f.write("#     " + detail + "\n")
        f.write(f"#   原因「源未收录」：{len(miss)} 个（语料中出现但三源均未收录，视同丢弃）\n")
        f.write("#     样例：" + " ".join(f"{c}({n})" for c, n in miss[:50]) + "\n")
        f.write(f"#   丢弃字符去重合计：{total_drop + len(miss)} 个\n#\n")

        f.write("# 【③ 被丢弃字中命中语料的 token Top20】\n")
        if miss:
            f.write("#   " + " ".join(f"{c}{n}" for c, n in miss[:20]) + "\n")
        else:
            f.write("#   无\n")
    return len(miss)


def main():
    ensure_sources()
    t2s = load_t2s()
    parsers = {
        "dockerian": parse_dockerian(),
        "charlesix59": parse_charlesix(),
        "weihethu": parse_weihethu(),
    }
    for tag, src in parsers.items():
        miss = sorted(set(CANON_KEYS) - set(src))
        n_uniq = len({c for v in src.values() for c in v})
        print(f"[源] {tag}: {len(src)} 组 / 去重 {n_uniq} 字" + (f"，缺 {miss}" if miss else ""))

    simp, extra, multi_hit = normalize_sources(parsers, t2s)
    accepted, char_groups, src_groups, support = cross_validate(simp)
    reasons = classify_conflicts(char_groups, src_groups, multi_hit)
    records = build_records(accepted)

    # 门禁：token 级覆盖率 < 95% 即中止（先报告后判门，未达标不落盘 JSON，避免误导性产物）
    cov = report_coverage(char_groups)
    if cov < MIN_COVERAGE:
        print(f"[BLOCKED] token 级覆盖率 {cov:.4%} < 门槛 {MIN_COVERAGE:.0%}，中止构建，"
              f"请人工复核（补表 / 退化为今音声调近似并显式标注）")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
        f.write("\n")

    tone_cnt = Counter(v["tone"] for v in records.values())
    print(f"[落盘] {OUT_JSON}：{len(records)} 韵部 | "
          + " ".join(f"{t}{tone_cnt[t]}" for t in ("平", "上", "去", "入"))
          + f" | 字条合计 {sum(len(v['chars']) for v in records.values())}"
          + f" | 去重字 {len(char_groups)}")

    n_conf = write_conflicts(reasons, src_groups)
    n_multi = sum(1 for rs in reasons.values() if "多韵部" in rs)
    n_cross = sum(1 for rs in reasons.values() if "跨源韵部不一致" in rs)
    print(f"[落盘] {OUT_CONFLICT}：冲突字 {n_conf} 个（多韵部 {n_multi} / 跨源不一致 {n_cross}）")

    n_drop = write_audit(support, extra, _coverage(char_groups)["miss"], reasons)
    print(f"[落盘] {OUT_AUDIT}：源未收录 {n_drop} 个；源流滤除非基本汉字区 "
          f"{sum(len(v) for v in extra.values())} 个")


if __name__ == "__main__":
    main()
