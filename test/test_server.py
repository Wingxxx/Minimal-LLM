"""推理服务测试：Tracker / Request / InferenceEngine / validate_params / create_server

分两层：零件级（脱网纯函数与组件）+ 集成级（起真实 HTTP 服务、真档上阵）。
运行：`python test/test_server.py`
"""
import sys, os, json, time, socket, struct, threading
import urllib.request, urllib.error, http.client
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import (Tracker, Request, InferenceEngine, validate_params, create_server,
                    load_serving_model, ARCH)
import train
import meter


# ---------- 夹具 ----------

class _StubModel:
    """GPT.generate 的替身：不碰权重，按序回吐 gen_ids[:max_new_tokens]。

    boom_at：第几次调用抛异常（None 表示永不抛），用于验证「单请求异常不杀推理线程」。
    delay：每吐一个 token 后 sleep 的秒数，用于给客户端制造「读到一半就跑」的时机。
    """

    def __init__(self, gen_ids=(1, 2, 3), boom_at=None, delay=0.0):
        self.gen_ids = list(gen_ids)
        self.boom_at = boom_at
        self.delay = delay
        self.calls = 0

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 kv_cache=None, logits_processor=None, on_token=None):
        self.calls += 1
        if self.boom_at == self.calls:
            raise RuntimeError("桩模型故意失败")
        picked = self.gen_ids[:max_new_tokens]
        for t in picked:
            if on_token is not None:
                on_token(int(t))
            if self.delay:
                time.sleep(self.delay)
        return np.concatenate([idx, np.asarray([picked], dtype=idx.dtype)], axis=1)

_ITOS = {1: "甲", 2: "乙", 3: "丙", 4: "丁"}
_STOI = {"床": 0, "前": 1, "明": 2, "月": 3, "光": 4, "，": 5, "。": 6, "\n": 7,
         "<五绝>": 8, "<七绝>": 9, "<五律>": 10, "<七律>": 11, "<杂言>": 12}
_CTX = 8

def _mkparams(prompt="床前明月光", max_tokens=3, **kw):
    p = {"prompt": prompt, "head": prompt, "prompt_ids": [0, 1, 2, 3, 4], "style": None,
         "max_tokens": max_tokens, "temperature": 1.0, "top_k": None, "top_p": None,
         "stream": False, "enforce_meter": False, "oov_dropped": 0, "logits_processor": None}
    p.update(kw)
    return p

def _body(prompt="床前明月光", **kw):
    d = {"prompt": prompt}
    d.update(kw)
    return json.dumps(d).encode("utf-8")

def _http(port, path, body=None, timeout=10):
    """发一次请求，返回 (状态码, 原始响应体)；4xx/5xx 不抛异常，便于逐一核对。"""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=body,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------- Step 2：Tracker ----------

def test_tracker_empty():
    """空缓冲返回 {}，绝不编造数字"""
    assert Tracker().summary() == {}
    print("Tracker empty PASS")

def test_tracker_count_and_avg():
    t = Tracker()
    for x in (10.0, 20.0, 30.0):
        t.record(x)
    s = t.summary()
    assert s["count"] == 3 and s["avg_ms"] == 20.0, s
    print("Tracker count/avg PASS", s)

def test_tracker_percentile():
    """最近秩法：1..100 → p50=50、p99=99"""
    t = Tracker()
    for x in range(1, 101):
        t.record(x)
    s = t.summary()
    assert s["avg_ms"] == 50.5 and s["p50_ms"] == 50 and s["p99_ms"] == 99, s
    print("Tracker percentile PASS", s)

def test_tracker_ring_buffer():
    """cap=3 记 5 条 → 只留最近 3 条（老元素被挤出）"""
    t = Tracker(cap=3)
    for x in (10.0, 20.0, 30.0, 40.0, 50.0):
        t.record(x)
    s = t.summary()
    assert s["count"] == 3 and s["avg_ms"] == 40.0, s
    print("Tracker ring buffer PASS", s)

def test_tracker_single_record():
    t = Tracker()
    t.record(7.5)
    s = t.summary()
    assert s["count"] == 1 and s["avg_ms"] == 7.5 and s["p50_ms"] == 7.5, s
    print("Tracker single PASS", s)

def test_tracker_thread_safe():
    """4 线程各记 50 条：锁有效，计数不丢"""
    t = Tracker(cap=1000)
    def worker():
        for _ in range(50):
            t.record(1.0)
    ths = [threading.Thread(target=worker) for _ in range(4)]
    for th in ths: th.start()
    for th in ths: th.join()
    assert t.summary()["count"] == 200, t.summary()
    print("Tracker thread-safe PASS")


# ---------- Step 3：Request / InferenceEngine ----------

def test_engine_two_requests_isolated():
    """两请求先后提交：结果正确且参数互不污染"""
    eng = InferenceEngine(_StubModel(), _ITOS, ctx_len=_CTX)
    eng.start()
    try:
        r1, r2 = Request(_mkparams("床前明月光")), Request(_mkparams("白日依山尽"))
        eng.submit(r1); eng.submit(r2)
        assert r1.done.wait(5) and r2.done.wait(5)
        assert r1.error is None and r2.error is None, (r1.error, r2.error)
        assert r1.result["prompt"] == "床前明月光" and r2.result["prompt"] == "白日依山尽"
        assert r1.result["text"] == "甲乙丙" and r2.result["text"] == "甲乙丙"
        assert r1.result["tokens"] == 3 and r1.result["elapsed_ms"] >= 0
    finally:
        eng.stop()
    print("Engine two requests PASS")

def test_engine_survives_error():
    """单请求异常只写 req.error，推理线程存活、后续请求照常"""
    eng = InferenceEngine(_StubModel(boom_at=1), _ITOS, ctx_len=_CTX)
    eng.start()
    try:
        r1 = Request(_mkparams()); eng.submit(r1)
        assert r1.done.wait(5)
        assert isinstance(r1.error, RuntimeError), r1.error
        assert eng._thread.is_alive(), "单请求异常杀死了推理线程"
        r2 = Request(_mkparams()); eng.submit(r2)
        assert r2.done.wait(5)
        assert r2.error is None and r2.result["tokens"] == 3, r2.error
    finally:
        eng.stop()
    print("Engine survives error PASS", type(r1.error).__name__)

def test_engine_cancelled():
    """cancelled=True：on_token 一次不调，但结果仍完整回填（不丢请求）"""
    eng = InferenceEngine(_StubModel(gen_ids=(1, 2, 3, 4)), _ITOS, ctx_len=_CTX)
    eng.start()
    try:
        seen = []
        r = Request(_mkparams(max_tokens=4), on_token=seen.append)
        r.cancelled = True
        eng.submit(r)
        assert r.done.wait(5)
        assert seen == [], seen
        assert r.result["text"] == "甲乙丙丁", r.result
    finally:
        eng.stop()
    print("Engine cancelled PASS", len(seen))

def test_engine_stream_order():
    """on_token 回调节奏与顺序：甲乙丙丁逐项"""
    eng = InferenceEngine(_StubModel(gen_ids=(1, 2, 3, 4)), _ITOS, ctx_len=_CTX)
    eng.start()
    try:
        seen = []
        r = Request(_mkparams(max_tokens=4), on_token=seen.append)
        eng.submit(r)
        assert r.done.wait(5)
        assert "".join(seen) == "甲乙丙丁", seen
    finally:
        eng.stop()
    print("Engine stream order PASS", "".join(seen))

def test_engine_stop():
    eng = InferenceEngine(_StubModel(), _ITOS, ctx_len=_CTX)
    eng.start()
    th = eng._thread
    eng.stop()
    th.join(5)
    assert not th.is_alive(), "stop() 后推理线程仍未退出"
    print("Engine stop PASS")


# ---------- Step 4：validate_params（7 类客户端错误 + 加固） ----------

def test_validate_defaults():
    # ctx 取 64：默认 max_tokens=32，用 _CTX=8 会先撞「超 ctx」而非验默认值
    p, err = validate_params(_body(), _STOI, 64)
    assert err is None, err
    assert p["max_tokens"] == 32 and p["temperature"] == 0.8, p
    assert p["top_k"] == 20 and p["top_p"] is None, p
    assert p["stream"] is False and p["enforce_meter"] is False
    assert p["logits_processor"] is None and p["style"] is None
    print("validate defaults PASS")

def test_validate_bad_json():
    for raw in (b"{oops", b"\xff\xfe", b"[1,2]"):
        p, err = validate_params(raw, _STOI, _CTX)
        assert p is None and err["status"] == 400 and err["error"] == "bad json", (raw, err)
    print("validate bad json PASS")

def test_validate_invalid_field_type():
    """类型加固：bool 不得冒充 int（True 是 int 子类），字符串 prompt 必须是 str"""
    for body in ({"prompt": "床前明月光", "max_tokens": True},
                 {"prompt": 123},
                 {"prompt": "床前明月光", "stream": "yes"}):
        p, err = validate_params(json.dumps(body).encode("utf-8"), _STOI, _CTX)
        assert p is None and err["error"] == "invalid field type", (body, err)
        assert err["field"] in body, err
    print("validate field type PASS")

def test_validate_invalid_style():
    p, err = validate_params(_body(style="唐诗"), _STOI, _CTX)
    assert p is None and err["status"] == 400 and err["error"] == "invalid style", err
    assert err["allowed"] == list(("五绝", "七绝", "五律", "七律", "杂言")), err
    print("validate invalid style PASS", err["allowed"])

def test_validate_empty_prompt():
    p, err = validate_params(_body(prompt=""), _STOI, _CTX)
    assert p is None and err["error"] == "empty prompt after tokenize", err
    print("validate empty prompt PASS")

def test_validate_all_oov_prompt():
    """全为词表外字符 → 分词后为空，同样响亮报错（严禁换行符兜底）"""
    p, err = validate_params(_body(prompt="XYZ"), _STOI, _CTX)
    assert p is None and err["error"] == "empty prompt after tokenize", err
    print("validate all-OOV prompt PASS")

def test_validate_prompt_filling_ctx():
    """prompt 已占满 ctx：明示还能写 0 个，而不是静默截断"""
    p, err = validate_params(_body(prompt="床前明月光床前明", max_tokens=1), _STOI, _CTX)
    assert p is None and err["error"] == "prompt too long", err
    assert err["prompt_tokens"] == 8 and err["max_tokens_available"] == 0, err
    print("validate prompt too long PASS", err["prompt_tokens"], err["max_tokens_available"])

def test_validate_max_tokens_zero():
    for v in (0, -3):
        p, err = validate_params(_body(max_tokens=v), _STOI, _CTX)
        assert p is None and err["error"] == "invalid max_tokens", (v, err)
    print("validate max_tokens PASS")

def test_validate_temperature_zero():
    for v in (0, -0.5):
        p, err = validate_params(_body(temperature=v), _STOI, _CTX)
        assert p is None and err["error"] == "invalid temperature", (v, err)
    print("validate temperature PASS")

def test_validate_ok_style_prefix():
    """诗体 token 拼在 prompt 之前；9 字符 → 6 个 id（诗体只计 1 格）"""
    p, err = validate_params(_body(style="五绝", max_tokens=2), _STOI, _CTX)
    assert err is None, err
    assert p["head"] == "<五绝>床前明月光", p["head"]
    assert len(p["prompt_ids"]) == 6, p["prompt_ids"]
    print("validate style prefix PASS", p["head"], len(p["prompt_ids"]))

def test_validate_oov():
    """词表外字符逐字符跳过，并如实回告被跳过数量"""
    p, err = validate_params(_body(prompt="床前X"), _STOI, 64)
    assert err is None, err
    assert p["oov_dropped"] == 1, p
    assert len(p["prompt_ids"]) == 2, p["prompt_ids"]
    print("validate oov PASS", p["oov_dropped"])

def test_validate_top_k_top_p_normalized():
    """top_k ≤ 0 / null 一律归为不启用；top_p 原样保留"""
    p, _ = validate_params(_body(top_k=0, top_p=0.9), _STOI, 64)
    assert p["top_k"] is None and p["top_p"] == 0.9, p
    p, _ = validate_params(_body(top_k=None), _STOI, 64)
    assert p["top_k"] is None, p
    p, _ = validate_params(_body(top_k=-2), _STOI, 64)
    assert p["top_k"] is None, p
    print("validate top_k/top_p PASS")


# ---------- Step 5C：enforce_meter 接线 ----------

_PSTOI = {"床": 1, "前": 2, "明": 3, "月": 4, "光": 5, "，": 6, "。": 7, "\n": 8}

def _mini_generate(stoi, ids, steps, processor):
    """极简自回归：每步「所有非汉字得分 10、汉字得分 0」，经 processor 掩码后取 argmax。

    模拟一个「总想提前断句」的坏模型——若句长约束没生效，它第一步就吐标点。
    argmax 同分取最小下标，故汉字一律取最小 id，结果完全确定。
    """
    ids = list(ids)
    vocab = max(stoi.values()) + 1
    for _ in range(steps):
        logits = np.zeros(vocab, dtype=np.float64)
        logits[0] = -1e9                          # 0 号不在本迷你词表内，永久排除
        for c, i in stoi.items():
            logits[i] = 0.0 if ("\u4e00" <= c <= "\u9fff") else 10.0
        if processor is not None:
            logits = processor(logits, np.asarray([ids]))
        ids.append(int(np.argmax(logits)))
    return ids

def test_enforce_meter_wiring_holds_line_len():
    """enforce_meter=true：须挂上句长处理器，且它真能按住「想提前断句」的坏模型"""
    p, err = validate_params(_body(prompt="床前明月光", max_tokens=6, enforce_meter=True),
                             _PSTOI, 16)
    assert err is None, err
    proc = p["logits_processor"]
    assert proc is not None, "enforce_meter=true 却未挂句长处理器（5C 未接线）"

    # prompt「床前明月光」已是完整五言半句 → 推断 L=5：
    # 先断句「，」→ 再满 5 个汉字 → 再断句「。」，共 7 步
    ids = _mini_generate(_PSTOI, p["prompt_ids"], 7, proc)
    assert ids == [1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1, 7], ids
    # 对照组：同一坏模型不挂处理器 → 第一步就吐「，」（非汉字同分取最小下标），
    # 证上面那份规矩确由处理器所赐，而非坏模型自己守规矩
    bare = _mini_generate(_PSTOI, p["prompt_ids"], 1, None)
    assert bare == [1, 2, 3, 4, 5, 6], bare

    # 词表缺句读 → 响亮失败（否则约束静默失效，用户以为格律已生效）
    no_punct = {c: i for c, i in _PSTOI.items() if c not in ("，", "。")}
    p2, err2 = validate_params(_body(prompt="床前明月光", max_tokens=3, enforce_meter=True),
                               no_punct, 16)
    assert p2 is None and err2["error"] == "meter constraint unavailable", (p2, err2)
    print("validate enforce_meter wiring PASS", ids[5:])


# ---------- Step 5A：create_server / Handler（非流式） ----------

def test_server_routes_and_non_stream():
    """真起服务（端口由系统分配）：/health、非流式生成、/stats、404 全通"""
    srv, engine, port = create_server(_StubModel(), _STOI, _ITOS, ctx_len=_CTX, port=0,
                                      model_path="stub.npz")
    try:
        st, raw = _http(port, "/health")
        h = json.loads(raw)
        assert st == 200 and h["status"] == "ok" and h["ctx_len"] == _CTX, (st, h)
        assert h["model"] == "stub.npz", h

        st, raw = _http(port, "/v1/completions", _body(max_tokens=3))
        d = json.loads(raw)
        assert st == 200, (st, raw)
        assert d["text"] == "甲乙丙" and d["tokens"] == 3, d
        assert d["head"] == "床前明月光" and d["queued_ms"] >= 0, d

        st, raw = _http(port, "/stats")
        s = json.loads(raw)
        assert st == 200 and s["count"] == 1, (st, s)      # 实例挂载接线通，计数记到本服务

        st, raw = _http(port, "/nope")
        assert st == 404 and json.loads(raw)["error"] == "not found", (st, raw)
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("Server routes + non-stream PASS port=", port)


# ---------- Step 5B：流式 SSE + 取消协议 ----------

def _sse_events(text):
    """把 SSE 响应体切成事件内容列表：块间以空行分隔，每块形如 `data: <内容>`。"""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            out.append(block[len("data:"):].strip())
    return out

def _sse_request(port, body):
    """发一次流式请求，返回 (状态码, Content-Type, 响应体文本)；Connection: close 故读到 EOF 即止。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    status, ctype = r.status, r.getheader("Content-Type")
    text = r.read().decode("utf-8")
    conn.close()
    return status, ctype, text

def test_server_stream_sse_events():
    """流式：200 + text/event-stream；逐 token 一条事件；[DONE] 收尾；拼起来等于非流式文本"""
    srv, engine, port = create_server(_StubModel(), _STOI, _ITOS, ctx_len=_CTX, port=0)
    try:
        st, raw = _http(port, "/v1/completions", _body(max_tokens=3))
        assert st == 200, (st, raw)
        baseline = json.loads(raw)["text"]                  # 同一模型、同一参数 → 应逐字相同

        st, ctype, text = _sse_request(port, _body(max_tokens=3, stream=True))
        assert st == 200, (st, text)
        assert ctype.startswith("text/event-stream"), ctype
        evs = _sse_events(text)
        assert evs[-1] == "[DONE]", evs                     # 收尾暗号
        toks = [json.loads(e)["token"] for e in evs[:-1]]
        assert len(toks) == 3, evs                          # 逐条吐，不许攒批成一条
        assert "".join(toks) == baseline, (toks, baseline)   # 吐的与给的必须一致
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("Server stream SSE PASS", "".join(toks))

def test_server_stream_client_abort_survives():
    """客户端读到一半就跑：写异常被捕获并插取消旗，服务端与推理线程照常

    必须直接断言「取消旗被插上」——否则即便写异常未被捕获，异常也会经推理线程
    主循环兜住，线程照样存活，用例会假绿。
    """
    submitted = []
    orig_submit = InferenceEngine.submit

    def spy_submit(self, req):
        submitted.append(req)
        orig_submit(self, req)

    InferenceEngine.submit = spy_submit
    srv = engine = None
    try:
        srv, engine, port = create_server(_StubModel(gen_ids=(1, 2, 3, 4), delay=0.15),
                                          _STOI, _ITOS, ctx_len=_CTX, port=0)
        body = json.dumps({"prompt": "床前", "max_tokens": 4,
                           "stream": True}).encode("utf-8")
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        head = ("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body))
        s.sendall(head.encode("utf-8") + body)
        buf = b""
        while b"data:" not in buf:                          # 读到首个事件才算「吐到一半」
            chunk = s.recv(4096)
            assert chunk, "服务端未推送任何 data 事件"
            buf += chunk
        assert b"200" in buf, buf
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()                                           # SO_LINGER=0 → 发 RST，服务端后续写必失败

        time.sleep(1.0)                                     # 等被弃请求在服务端走完
        assert submitted[0].cancelled is True, "客户端断开未触发取消（写异常未被捕获）"
        assert engine._thread.is_alive(), "推理线程被客户端断开拖死了"
        st, raw = _http(port, "/v1/completions", _body(prompt="床前", max_tokens=4))
        assert st == 200 and json.loads(raw)["tokens"] == 4, (st, raw)
        st, raw = _http(port, "/health")
        assert st == 200, (st, raw)
    finally:
        InferenceEngine.submit = orig_submit
        if srv is not None:
            srv.shutdown(); engine.stop(); srv.server_close()
    print("Server stream client abort PASS")


# ---------- Step 6：集成测试（真档上阵 + 真 HTTP 全链路） ----------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT = os.path.join(_ROOT, "model-gelv-m3.npz")
_real_cache = {}

def _real():
    """载入真实权重档（进程内只载一次：载档与首次前向较慢，重复载纯属浪费）。"""
    if "m" not in _real_cache:
        _real_cache["m"] = load_serving_model(_CKPT)
    return _real_cache["m"]

def _hanzi_count(s):
    """汉字个数（句长判据用；标点、换行、诗体标签一律不计）。"""
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")

def test_real_archive_matches_arch():
    """真档形状与 ARCH / 服务端词表三方交叉校验：防骨干常量漂移与载错档"""
    model, stoi, itos = _real()
    d = np.load(_CKPT)
    assert d["tok_emb"].shape == (len(stoi), ARCH["d_model"]), d["tok_emb"].shape
    assert d["pos_emb"].shape == (ARCH["ctx_len"], ARCH["d_model"]), d["pos_emb"].shape
    # 只数模型参数：档内还存着 _opt.m / _opt.v 优化器状态（同名前缀），不加 startswith 会数成三倍
    n_blocks = sum(1 for k in d.files if k.startswith("blocks.") and k.endswith(".ln1.gamma"))
    assert n_blocks == ARCH["n_layer"], n_blocks
    assert len(d["_chars"]) + len(train.SPECIAL_TOKENS) == len(stoi), len(d["_chars"])
    assert all(stoi[ch] == i for i, ch in enumerate(d["_chars"].tolist())), \
        "档内词表顺序与服务端 stoi 不一致（id↔字映射会错位，生成必然乱码）"
    assert (model.d_model, model.n_head, model.n_layer, model.ctx_len) == (
        ARCH["d_model"], ARCH["n_head"], ARCH["n_layer"], ARCH["ctx_len"])
    print("real archive vs ARCH PASS", d["tok_emb"].shape, "vocab", len(stoi))

def test_real_generate_non_stream_and_stream():
    """真档非流式与流式：同种子下流式拼接逐字等于非流式文本"""
    model, stoi, itos = _real()
    srv, engine, port = create_server(model, stoi, itos, ctx_len=ARCH["ctx_len"],
                                      port=0, model_path=_CKPT)
    try:
        body = _body(max_tokens=8)
        np.random.seed(20260920)                     # 采样走全局 numpy 随机流，同种子即同结果
        st, raw = _http(port, "/v1/completions", body)
        assert st == 200, (st, raw)
        d = json.loads(raw)
        assert d["text"] and 0 < d["tokens"] <= 8, d
        assert d["elapsed_ms"] > 0 and d["queued_ms"] >= 0, d
        assert d["head"] == "床前明月光", d

        np.random.seed(20260920)                     # 复位同一采样流再走流式
        st2, ctype, text = _sse_request(port, _body(max_tokens=8, stream=True))
        assert st2 == 200 and ctype.startswith("text/event-stream"), (st2, ctype)
        evs = _sse_events(text)
        assert evs[-1] == "[DONE]", evs
        toks = [json.loads(e)["token"] for e in evs[:-1]]
        assert len(toks) == d["tokens"], (toks, d)
        assert "".join(toks) == d["text"], (toks, d["text"])
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("real non-stream + stream PASS", repr(d["text"]))

def test_real_style_prefix_reaches_model():
    """真档 + style=五绝：回告 head 以 <五绝> 起头，即模型实际所见输入"""
    model, stoi, itos = _real()
    srv, engine, port = create_server(model, stoi, itos, ctx_len=ARCH["ctx_len"], port=0)
    try:
        np.random.seed(11)
        st, raw = _http(port, "/v1/completions", _body(style="五绝", max_tokens=3))
        assert st == 200, (st, raw)
        d = json.loads(raw)
        assert d["head"].startswith("<五绝>"), d
        assert d["head"] == "<五绝>床前明月光" and d["style"] == "五绝", d
        assert d["text"] and d["tokens"] == 3, d
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("real style prefix PASS", d["head"])

def test_real_enforce_meter_tail_anchor():
    """真档 + enforce_meter：满半句即断、恰 5 汉字，末半句经 infer_line_len 判定仍为 5（尾锚）"""
    model, stoi, itos = _real()
    srv, engine, port = create_server(model, stoi, itos, ctx_len=ARCH["ctx_len"], port=0)
    try:
        np.random.seed(7)
        st, raw = _http(port, "/v1/completions", _body(max_tokens=7, enforce_meter=True))
        assert st == 200, (st, raw)
        d = json.loads(raw)
        t = d["text"]
        assert d["tokens"] == 7, d
        assert t[0] == "，" and t[-1] == "。", t            # 半句满 5 即断，且断在目标标点
        assert _hanzi_count(t) == 5, t                     # 中间恰 5 个汉字，多一个少一个都不行
        ids = train.encode(t, stoi)                        # 尾锚判据：从生成文本反推句长
        stop = [i for c, i in stoi.items() if not ("\u4e00" <= c <= "\u9fff")]
        assert meter.infer_line_len(ids, stop) == 5, (t, ids)
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("real enforce_meter tail anchor PASS", repr(d["text"]))

def test_real_http_client_errors():
    """7 类客户端错误走真 HTTP 链路：状态码与 error 字段逐一核对"""
    srv, engine, port = create_server(_StubModel(), _STOI, _ITOS, ctx_len=_CTX, port=0)
    cases = (
        ("路径不匹配", "/nope", None, 404, "not found"),
        ("非法 JSON", "/v1/completions", b"{oops", 400, "bad json"),
        ("非法诗体", "/v1/completions", _body(style="唐诗"), 400, "invalid style"),
        ("空 prompt", "/v1/completions", _body(prompt="abc"), 400, "empty prompt after tokenize"),
        ("超 ctx_len", "/v1/completions", _body(max_tokens=32), 400, "prompt too long"),
        ("非法 max_tokens", "/v1/completions", _body(max_tokens=0), 400, "invalid max_tokens"),
        ("非法 temperature", "/v1/completions", _body(temperature=0), 400, "invalid temperature"),
    )
    try:
        for name, path, body, want_status, want_err in cases:
            st, raw = _http(port, path, body)
            assert st == want_status, (name, st, raw)
            assert json.loads(raw)["error"] == want_err, (name, raw)
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("real HTTP client errors PASS", len(cases), "类")

def test_server_500_on_model_error():
    """模型抛异常：回 500 + 异常类名，且推理线程存活、后续请求照常"""
    srv, engine, port = create_server(_StubModel(boom_at=1), _STOI, _ITOS,
                                      ctx_len=_CTX, port=0)
    try:
        st, raw = _http(port, "/v1/completions", _body(max_tokens=3))
        assert st == 500, (st, raw)
        e = json.loads(raw)
        assert e["error"] == "inference failed" and e["type"] == "RuntimeError", e
        assert engine._thread.is_alive(), "单次推理异常不得杀死推理线程"
        st, raw = _http(port, "/v1/completions", _body(max_tokens=3))
        assert st == 200 and json.loads(raw)["tokens"] == 3, (st, raw)
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("Server 500 on model error PASS")

def test_server_concurrent_four_requests():
    """4 线程并发：全部 200、各自回显正确、模型恰被调用 4 次、/stats 计数恰为 4"""
    stub = _StubModel()
    srv, engine, port = create_server(stub, _STOI, _ITOS, ctx_len=_CTX, port=0)
    prompts = ("床前明月光", "明月光床前", "月光床前明", "光床前明月")
    results, errors = {}, []

    def worker(i):
        try:
            results[i] = _http(port, "/v1/completions",
                               _body(prompt=prompts[i], max_tokens=3))
        except Exception as e:                       # 线程内异常不落到主线程外
            errors.append((i, e))

    try:
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        assert not errors, errors
        assert len(results) == 4, results
        for i in range(4):
            st, raw = results[i]
            assert st == 200, (i, st, raw)
            d = json.loads(raw)
            assert d["prompt"] == prompts[i], (i, d)         # 各自回显，无串台
            assert d["text"] == "甲乙丙" and d["tokens"] == 3, (i, d)
        assert stub.calls == 4, stub.calls                    # 请求不丢不重
        st, raw = _http(port, "/stats")
        s = json.loads(raw)
        assert st == 200 and s["count"] == 4, (st, s)          # 计数恰为本次 4 笔
    finally:
        srv.shutdown(); engine.stop(); srv.server_close()
    print("Server concurrent 4 PASS")

def test_server_504_on_slow_model():
    """慢模型 + 极小 timeout：回 504，被弃请求仍被算完（非丢弃），推理线程存活（放最末）"""
    submitted = []
    orig_submit = InferenceEngine.submit

    def spy_submit(self, req):
        submitted.append(req)
        orig_submit(self, req)

    InferenceEngine.submit = spy_submit
    srv = engine = None
    try:
        srv, engine, port = create_server(_StubModel(delay=0.5), _STOI, _ITOS,
                                          ctx_len=_CTX, port=0, timeout=0.3)
        st, raw = _http(port, "/v1/completions", _body(max_tokens=3))
        assert st == 504, (st, raw)
        assert json.loads(raw)["error"] == "inference timeout", raw
        assert engine._thread.is_alive(), "超时取消不得杀死推理线程"
        time.sleep(1.8)                          # 等慢请求在服务端跑完（3 × 0.5s）
        assert submitted[0].cancelled is True, "超时未插取消旗"
        assert submitted[0].result is not None, "被弃请求被丢弃了（应照常算完，只是不回写 socket）"
    finally:
        InferenceEngine.submit = orig_submit
        if srv is not None:
            srv.shutdown(); engine.stop(); srv.server_close()
    print("Server 504 on slow model PASS")


if __name__ == "__main__":
    test_tracker_empty(); test_tracker_count_and_avg(); test_tracker_percentile()
    test_tracker_ring_buffer(); test_tracker_single_record(); test_tracker_thread_safe()
    test_engine_two_requests_isolated(); test_engine_survives_error(); test_engine_cancelled()
    test_engine_stream_order(); test_engine_stop()
    test_validate_defaults(); test_validate_bad_json(); test_validate_invalid_field_type()
    test_validate_invalid_style(); test_validate_empty_prompt(); test_validate_all_oov_prompt()
    test_validate_prompt_filling_ctx(); test_validate_max_tokens_zero()
    test_validate_temperature_zero(); test_validate_ok_style_prefix(); test_validate_oov()
    test_validate_top_k_top_p_normalized()
    test_enforce_meter_wiring_holds_line_len()
    test_server_routes_and_non_stream()
    test_server_stream_sse_events(); test_server_stream_client_abort_survives()
    test_real_archive_matches_arch()
    test_real_generate_non_stream_and_stream()
    test_real_style_prefix_reaches_model()
    test_real_enforce_meter_tail_anchor()
    test_real_http_client_errors()
    test_server_500_on_model_error()
    test_server_concurrent_four_requests()
    test_server_504_on_slow_model()          # 慢桩会占住推理线程，务必放最后
    print("ALL SERVER TESTS PASS")
