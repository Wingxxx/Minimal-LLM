"""HTTP 推理服务：单推理线程串行调度 + SSE 流式 + 延迟统计

架构约束（PLAN §6.1）：
  - `GPT.forward` 计算途中会把中间量写入实例属性（KV Cache、self.logits 等），
    多线程并发调用会互相覆盖，构成数据竞争。故**模型只有一个主人**：全局唯一的
    推理线程驱动，HTTP 线程仅入队/等待，无需对模型加锁。
  - HTTP 线程与推理线程不直接对话，只经 `Request` 托盘交接：params 进、
    result/error 出、`threading.Event` 作完成信号灯。
  - 超时或客户端断开时只插 `cancelled` 旗，**绝不**让异常杀死唯一的推理线程（§6.3）。

启动：`python server.py --model model-gelv-m3.npz --port 8000`
"""
import argparse
import json
import math
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import meter
import train
from model.backend import np as onp, as_numpy
from model.gpt import GPT

# 骨干常量（与权重档逐项匹配；测试按档内形状交叉校验，防常量漂移）
ARCH = dict(d_model=256, n_head=8, n_layer=6, ctx_len=128)
# 合法诗体：由特殊 token 派生（"<五绝>" -> "五绝"），避免两处定义漂移
STYLES = tuple(tok[1:-1] for tok in train.SPECIAL_TOKENS)
# 请求字段类型表；bool 单独处理（bool 是 int 的子类，混用会放过 True 当 max_tokens）
_FIELD_TYPES = {
    "prompt": (str,),
    "style": (str,),
    "max_tokens": (int,),
    "temperature": (int, float),
    "top_k": (int,),
    "top_p": (int, float),
    "stream": (bool,),
    "enforce_meter": (bool,),
}
_FLAG_FIELDS = ("stream", "enforce_meter")


class Tracker:
    """请求延迟统计：环形缓冲只留最近 cap 条，避免长跑服务内存无界增长。"""

    def __init__(self, cap=200):
        self._xs = deque(maxlen=int(cap))     # 多 HTTP 线程并发写，需加锁
        self._lock = threading.Lock()

    def record(self, dt_ms):
        with self._lock:
            self._xs.append(float(dt_ms))

    def summary(self):
        with self._lock:
            xs = sorted(self._xs)
        if not xs:                            # 无数据时返回空对象，绝不编造数字
            return {}
        n = len(xs)
        return {"count": n,
                "avg_ms": round(sum(xs) / n, 3),
                "p50_ms": round(self._pct(xs, 50), 3),
                "p99_ms": round(self._pct(xs, 99), 3)}

    @staticmethod
    def _pct(xs, p):
        """最近秩法求百分位：下标 = ceil(p/100 * n) - 1（xs 已升序）。"""
        k = max(1, math.ceil(p / 100.0 * len(xs))) - 1
        return xs[min(k, len(xs) - 1)]


class Request:
    """一次推理请求的载体：参数进、结果/异常出、Event 作完成信号。"""

    def __init__(self, params, on_token=None):
        self.params = params
        self.on_token = on_token
        self.result = None
        self.error = None
        self.done = threading.Event()
        self.cancelled = False            # 客户端断开/超时后置真：回调不再触碰 socket
        self.enqueued_at = 0.0            # 入队时刻（perf_counter），用于算排队时长


class InferenceEngine:
    """模型唯一主人：串行消费请求队列，独占调用 model.generate。"""

    def __init__(self, model, itos, ctx_len=ARCH["ctx_len"]):
        self.model = model
        self.itos = itos
        self.ctx_len = ctx_len
        self._q = queue.Queue()
        self._thread = None

    def submit(self, req):
        req.enqueued_at = time.perf_counter()
        self._q.put(req)

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        """投毒丸（None）+ join：排队中的请求会被先跑完，随后线程退出。"""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout)
        self._thread = None

    def _loop(self):
        while True:
            req = self._q.get()
            if req is None:
                return
            try:
                req.result = self._run(req)
            except Exception as exc:      # 单请求异常一律兜住：绝不终止推理线程
                req.error = exc
            finally:
                req.done.set()

    def _run(self, req):
        p = req.params
        started = time.perf_counter()
        queued_ms = (started - req.enqueued_at) * 1000.0
        ids = onp.asarray([p["prompt_ids"]], dtype=onp.int64)
        cb = None
        if req.on_token is not None:
            def cb(tok):
                # 取消后不再回调：连接已断，写 socket 只会抛异常
                if not req.cancelled:
                    req.on_token(self.itos.get(tok, ""))
        out = self.model.generate(ids, p["max_tokens"], temperature=p["temperature"],
                                  top_k=p["top_k"], top_p=p["top_p"],
                                  logits_processor=p["logits_processor"], on_token=cb)
        new = as_numpy(out)[0, ids.shape[1]:]          # 只取续写部分（不含 prompt）
        text = "".join(self.itos.get(int(t), "") for t in new)
        result = {"text": text,
                  "prompt": p["prompt"],
                  "style": p["style"],
                  "head": p["head"],
                  "tokens": int(len(new)),
                  "queued_ms": round(queued_ms, 2),
                  "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2)}
        if p["oov_dropped"] > 0:                        # 仅当确有字符被跳过时才出现
            result["oov_dropped"] = int(p["oov_dropped"])
        return result


def validate_params(raw_body, stoi, ctx_len):
    """校验请求体并整理成推理参数；返回 (params, err)，err 非空即为客户端/配置错误。

    收**原始字节**而非已解析对象：让「非合法 JSON / 非法 UTF-8」也落在纯函数内，
    从而可脱网穷举全部 7 类客户端错误。全部失败一律**响亮报错**，严禁静默降级。
    """
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, {"status": 400, "error": "bad json"}
    if not isinstance(body, dict):
        return None, {"status": 400, "error": "bad json"}

    for field, types in _FIELD_TYPES.items():
        if field not in body or body[field] is None:
            continue
        v = body[field]
        if field in _FLAG_FIELDS:
            ok = isinstance(v, bool)
        else:
            ok = not isinstance(v, bool) and isinstance(v, types)
        if not ok:
            return None, {"status": 400, "error": "invalid field type", "field": field}

    style = body.get("style")
    if style is not None and style not in STYLES:
        return None, {"status": 400, "error": "invalid style", "allowed": list(STYLES)}

    prompt = body.get("prompt") or ""
    if not prompt:
        return None, {"status": 400, "error": "empty prompt after tokenize"}
    head = ("<%s>" % style) + prompt if style else prompt
    ids = train.encode(head, stoi)
    if len(ids) == 0:      # 全为词表外字符：严禁以换行符兜底，否则掩盖客户端输入错误
        return None, {"status": 400, "error": "empty prompt after tokenize"}

    max_tokens = body.get("max_tokens", 32)
    if max_tokens is None:
        max_tokens = 32
    if int(max_tokens) <= 0:
        return None, {"status": 400, "error": "invalid max_tokens"}
    temperature = body.get("temperature", 0.8)
    if temperature is None:
        temperature = 0.8
    if float(temperature) <= 0:
        return None, {"status": 400, "error": "invalid temperature"}
    top_k = body.get("top_k", 20)
    top_p = body.get("top_p")

    available = ctx_len - len(ids)
    if available < max_tokens:
        return None, {"status": 400, "error": "prompt too long", "prompt_tokens": len(ids),
                      "ctx_len": ctx_len, "max_tokens_available": max(available, 0)}

    enforce_meter = bool(body.get("enforce_meter", False))
    processor = _meter_processor(stoi) if enforce_meter else None
    if enforce_meter and processor is None:
        # 词表缺句读 → 句长约束无法执行。响亮失败，严禁当作「无约束」照常生成。
        return None, {"status": 500, "error": "meter constraint unavailable",
                      "detail": "服务端词表缺少中文句读「，」「。」，无法执行句长约束"}

    # OOV 计数：每个诗体 token 在文本占 4 字符、在 id 流只计 1 个，故补回 3 字符/个
    n_tag = sum(head.count(tok) for tok in train.SPECIAL_TOKENS if tok in stoi)
    oov_dropped = len(head) - len(ids) - 3 * n_tag

    return {
        "prompt": prompt,
        "head": head,
        "prompt_ids": [int(t) for t in ids],
        "style": style,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_k": None if (top_k is None or int(top_k) <= 0) else int(top_k),
        "top_p": None if top_p is None else float(top_p),
        "stream": bool(body.get("stream", False)),
        "enforce_meter": enforce_meter,
        "oov_dropped": oov_dropped,
        "logits_processor": processor,
    }, None


def _meter_processor(stoi):
    """构造本请求专属的句长约束处理器；词表缺句读时返回 None（由调用方响亮拒绝）。

    口径与 tools/prosody_eval.py 一致：
      - punct_ids     「，」「。」（可作断句的标点）
      - non_hanzi_ids 词表**全部**非汉字：既定义「什么不算句长」，也在未满句长时被整体屏蔽。
                      必须全量——漏掉标点会让标点被误计成汉字、约束静默失效
      - newline_id    换行符（「。」断句后放行一次）；用 `is None` 显式判定，
                      因换行 id 恰为 0，任何 `or` 写法都会把它当假值吞掉

    每请求新建一个：处理器把推断出的句长 L 冻结在闭包里，共用会串台。
    """
    punct_ids = [stoi[c] for c in ("，", "。") if c in stoi]
    if len(punct_ids) != 2:
        return None
    return meter.make_meter_processor(
        punct_ids, stoi["，"], stoi["。"],
        non_hanzi_ids=[i for c, i in stoi.items() if not ("\u4e00" <= c <= "\u9fff")],
        newline_id=stoi.get("\n"))


def load_serving_model(path):
    """载入推理权重档，返回 (model, stoi, itos)。

    词表按当前语料重建（`train.load_corpus()`），辅助头配置由档内参数自动识别
    （`rhyme_head.W` 列数定 n_rhyme、有无 `tone_head.W` 定 use_tone）。按名载参后
    断言无缺失参数——配置与档不符必须报错，绝不能静默保留随机初始化。
    """
    d = onp.load(path)
    names = set(d.files)
    n_rhyme = int(d["rhyme_head.W"].shape[1]) if "rhyme_head.W" in names else 0
    use_tone = "tone_head.W" in names
    _text, _chars, stoi, itos, _data, _special = train.load_corpus()
    model = GPT(vocab_size=len(stoi), n_rhyme=n_rhyme, use_tone=use_tone, **ARCH)
    missing = train.load_params_partial(path, model)
    assert missing == 0, f"{path} 有 {missing} 个参数未载入（辅助头配置与档不符），拒绝服务"
    return model, stoi, itos


class Handler(BaseHTTPRequestHandler):
    """HTTP 路由、参数校验、SSE 写回。

    engine/tracker/stoi 一律经 `self.server.*` 取——**严禁类属性**：类属性全进程共用，
    起第二个服务即覆盖第一个，测试串台。
    """

    protocol_version = "HTTP/1.1"

    # ---------- 路由 ----------

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/stats":
            self._send_json(200, self.server.tracker.summary())
        elif path == "/health":
            self._send_json(200, {"status": "ok", "model": self.server.model_path,
                                  "ctx_len": self.server.ctx_len})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/v1/completions":
            self._send_json(404, {"error": "not found"})
            return
        params, err = validate_params(self._read_body(), self.server.stoi, self.server.ctx_len)
        if err is not None:
            status = err["status"]
            self._send_json(status, {k: v for k, v in err.items() if k != "status"})
            return
        if params["stream"]:
            self._stream_completion(params)
            return

        req = Request(params)
        self.server.engine.submit(req)
        if not req.done.wait(self.server.timeout):
            req.cancelled = True            # 先插旗、再写 504：防推理线程继续写已断开的连接
            self._send_json(504, {"error": "inference timeout"})
            return
        if req.error is not None:
            self._send_json(500, {"error": "inference failed",
                                  "type": type(req.error).__name__,
                                  "detail": str(req.error)})
            return
        self.server.tracker.record(req.result["elapsed_ms"])
        self._send_json(200, req.result)

    # ---------- 流式（SSE） ----------

    def _stream_completion(self, params):
        """流式生成：先发 SSE 响应头，再逐 token 推送，末尾以 [DONE] 收尾。

        响应头一旦发出，状态码即锁死——此后任何异常都不能再改状态码（HTTP 协议使然），
        只能以一条 data: {"error": ...} 事件如实告知并立即收尾；故流式下的超时不回 504，
        改为在事件流内报错（非流式超时仍回 504）。

        取消协议（§6.3）：客户端半路断开时写 socket 会抛 BrokenPipeError /
        ConnectionAbortedError / OSError，捕获后置 req.cancelled = True 并停止写，
        推理线程据此不再回调——绝不让唯一的推理线程被已断开的连接拖垮。

        已知限制：on_token 在推理线程里直接写 socket，与超时收尾路径共用同一 wfile，
        故用 _sse_lock 串行化两处写入，避免事件行交错。
        """
        self._sse_lock = threading.Lock()
        self.close_connection = True                    # 无 Content-Length，以断连标记响应结束
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")         # send_header 会同步置 close_connection
        self.end_headers()

        req = Request(params)
        req.on_token = lambda ch: self._sse_token(req, ch)
        deadline = time.time() + self.server.timeout
        self.server.engine.submit(req)
        while not req.done.wait(0.2):
            if time.time() > deadline:
                req.cancelled = True                    # 先插旗：推理线程立即停止回调
                self._sse_send({"error": "inference timeout"})
                self._sse_raw("[DONE]")
                return
        if req.error is not None:
            self._sse_send({"error": "inference failed",
                            "type": type(req.error).__name__,
                            "detail": str(req.error)})
        elif not req.cancelled:
            self.server.tracker.record(req.result["elapsed_ms"])
        self._sse_raw("[DONE]")

    def _sse_token(self, req, ch):
        """推理线程回调：推一个 token 事件；连接已断则插旗令后续回调全部跳过。"""
        if req.cancelled:
            return
        if not self._sse_send({"token": ch}):
            req.cancelled = True

    def _sse_send(self, obj):
        """按 JSON 编码推一条 SSE 事件，返回 False 表示连接已断。"""
        return self._sse_raw(json.dumps(obj, ensure_ascii=False))

    def _sse_raw(self, text):
        """裸推一条 SSE 事件（text 原样作 data 内容）并立即 flush，返回 False 表示连接已断。

        flush 不可省：不刷新则事件攒在缓冲区里一并发出，流式退化为一次性返回。
        写入异常是本服务唯一允许静默的场景——客户端已断开，报了也无人可收；
        但必须返回 False 让调用方插旗停止，保住推理线程（§6.3）。
        """
        payload = ("data: %s\n\n" % text).encode("utf-8")
        try:
            with self._sse_lock:
                self.wfile.write(payload)
                self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, OSError):
            return False

    # ---------- 基础设施 ----------

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        return self.rfile.read(n) if n > 0 else b""

    def _send_json(self, status, obj):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def create_server(model, stoi, itos, ctx_len=ARCH["ctx_len"], host="127.0.0.1", port=0,
                  timeout=60.0, cap=200, model_path=""):
    """装配 Engine + Tracker + ThreadingHTTPServer，返回 (srv, engine, port)。

    port=0 由系统分配空闲端口（测试用），实际端口从 srv.server_address 读回。
    服务在后台 daemon 线程里 serve_forever，调用方负责 shutdown()/server_close()。
    """
    engine = InferenceEngine(model, itos, ctx_len=ctx_len)
    engine.start()
    tracker = Tracker(cap=cap)
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    # 实例挂载（严禁类属性）：每个服务各持自己的一套，多实例不串台
    srv.engine = engine
    srv.tracker = tracker
    srv.stoi = stoi
    srv.ctx_len = ctx_len
    srv.timeout = timeout
    srv.model_path = model_path
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    return srv, engine, srv.server_address[1]


def main(argv=None):
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="极简 LLM 推理服务")
    ap.add_argument("--model", default=os.path.join(base, "model-gelv-m3.npz"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=60.0, help="单请求等待上限（秒）")
    ap.add_argument("--cap", type=int, default=200, help="/stats 环形缓冲容量")
    args = ap.parse_args(argv)

    model, stoi, itos = load_serving_model(args.model)
    srv, engine, port = create_server(model, stoi, itos, ARCH["ctx_len"], host=args.host,
                                      port=args.port, timeout=args.timeout, cap=args.cap,
                                      model_path=args.model)
    print(f"serving on http://{args.host}:{port}  model={args.model}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        srv.shutdown()
        srv.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
