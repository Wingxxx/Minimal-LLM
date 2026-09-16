"""GPU 计算后端（model/backend.py）单测：缓存重定向、运行期探活回退、GPU 冒烟。

覆盖对象：model/backend.py
  1. 缓存重定向：CUPY_CACHE_DIR / CUDA_CACHE_PATH 必须指向项目内 _probe/（绕开沙箱拦截）
  2. setdefault 语义：用户显式设置缓存目录时不得被覆盖
  3. CPU 默认：未置 MINIMAL_GPU 时 np 为原生 numpy
  4. GPU 启用：置 MINIMAL_GPU=1 且运行期探活通过时 np 为 cupy
  5. 运行期探活：cupy 可导入但运行时运算抛异常时，np 必须回退原生 numpy
  6. 运行期探活同步：探活必须把结果拷回 host 强制设备同步，异步 runtime 错误不得漏判
  7. GPU 冒烟：小模型在 GPU 上跑 20 步 forward/backward/AdamW，loss 有限且编译缓存可写

实现说明：后端在 import 期即决定 np，故本测试用 importlib.reload 配合临时环境变量
分别验证各条路径；每个用例结束都重载回干净状态，避免用例间相互污染。测试仅依赖
标准库与 numpy，无第三方测试框架；GPU 冒烟在子进程中执行以隔离显存与模块状态，
GPU 不可用时打印「跳过（GPU 不可用）」而非失败。
"""
import contextlib
import importlib
import os
import subprocess
import sys
import types

import numpy as onp

# 将项目根目录加入模块搜索路径，使 `import model.backend` 在任意目录下均可生效
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model.backend as backend

PROBE_DIR = os.path.join(ROOT, "_probe")
PROBE_CUPY_DIR = os.path.join(PROBE_DIR, "cupy_cache")
PROBE_CUDA_DIR = os.path.join(PROBE_DIR, "cuda_cache")


def _norm(path):
    """归一化路径用于比较：Windows 下大小写与分隔符不敏感。"""
    return os.path.normcase(os.path.abspath(path))


@contextlib.contextmanager
def _env(**updates):
    """临时设置环境变量（值为 None 表示删除该变量），退出时恢复原值。"""
    saved = {}
    for key, val in updates.items():
        saved[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def test_cache_dirs_redirected():
    """缓存目录必须重定向到项目 _probe/ 下：清空环境变量后重载应自动 setdefault。"""
    with _env(MINIMAL_GPU=None, CUPY_CACHE_DIR=None, CUDA_CACHE_PATH=None):
        importlib.reload(backend)
        got_cupy = os.environ.get("CUPY_CACHE_DIR")
        got_cuda = os.environ.get("CUDA_CACHE_PATH")
        assert got_cupy is not None, "重载后 CUPY_CACHE_DIR 未设置"
        assert got_cuda is not None, "重载后 CUDA_CACHE_PATH 未设置"
        assert _norm(got_cupy) == _norm(PROBE_CUPY_DIR), f"CUPY_CACHE_DIR 未指向 _probe: {got_cupy}"
        assert _norm(got_cuda) == _norm(PROBE_CUDA_DIR), f"CUDA_CACHE_PATH 未指向 _probe: {got_cuda}"
        # 两者必须均落在项目 _probe/ 目录之下（防路径拼错）
        probe_norm = _norm(PROBE_DIR) + os.sep
        assert _norm(got_cupy).startswith(probe_norm), f"CUPY_CACHE_DIR 不在 _probe/ 下: {got_cupy}"
        assert _norm(got_cuda).startswith(probe_norm), f"CUDA_CACHE_PATH 不在 _probe/ 下: {got_cuda}"
    importlib.reload(backend)  # 复原为外部环境决定的干净状态
    print("cache_dirs_redirected PASS")


def test_setdefault_keeps_user_value():
    """setdefault 语义：用户显式设置的缓存目录不得被默认值覆盖。"""
    custom_cupy = os.path.join(PROBE_DIR, "custom_cupy")
    custom_cuda = os.path.join(PROBE_DIR, "custom_cuda")
    with _env(MINIMAL_GPU=None, CUPY_CACHE_DIR=custom_cupy, CUDA_CACHE_PATH=custom_cuda):
        importlib.reload(backend)
        assert _norm(os.environ["CUPY_CACHE_DIR"]) == _norm(custom_cupy), "CUPY_CACHE_DIR 被默认值覆盖"
        assert _norm(os.environ["CUDA_CACHE_PATH"]) == _norm(custom_cuda), "CUDA_CACHE_PATH 被默认值覆盖"
    importlib.reload(backend)
    print("setdefault_keeps_user_value PASS")


def test_cpu_default_backend():
    """未置 MINIMAL_GPU 时：USE_GPU 为 False 且 np 为原生 numpy。"""
    with _env(MINIMAL_GPU=None, CUPY_CACHE_DIR=None, CUDA_CACHE_PATH=None):
        be = importlib.reload(backend)
        use_gpu, np_mod = be.USE_GPU, be.np
    importlib.reload(backend)
    assert use_gpu is False, "未置 MINIMAL_GPU 时 USE_GPU 应为 False"
    assert np_mod is onp, "未置 MINIMAL_GPU 时 np 应为原生 numpy"
    print("cpu_default_backend PASS")


def test_gpu_enabled_backend():
    """置 MINIMAL_GPU=1 且运行期探活通过时：np 为 cupy 模块（GPU 不可用则跳过）。"""
    with _env(MINIMAL_GPU="1", CUPY_CACHE_DIR=None, CUDA_CACHE_PATH=None):
        be = importlib.reload(backend)
        cupy_mod, use_gpu, np_mod = be._cupy, be.USE_GPU, be.np
    importlib.reload(backend)
    if cupy_mod is None:
        print("跳过（cupy 未安装）")
        return
    if not use_gpu:
        print("跳过（GPU 运行期不可用，探活已回退 CPU）")
        return
    assert np_mod is cupy_mod, "MINIMAL_GPU=1 且探活通过时 np 应为 cupy"
    assert np_mod is not onp, "GPU 路径下 np 不应为题述 numpy"
    print("gpu_enabled_backend PASS")


def test_runtime_probe_fallback():
    """cupy 可导入但运行期运算抛异常时：USE_GPU 为 False 且 np 回退原生 numpy。

    注入桩模块模拟「已装 cupy、但 CUDA 驱动/runtime 故障」：zeros(...) 返回的对象
    在 sum()（触发真实 kernel 运算）时抛 RuntimeError。仅捕获 ImportError 的
    旧实现会漏掉此类故障并错把 np 指向不可用的 cupy。
    """
    real_cupy = sys.modules.get("cupy")
    stub = types.ModuleType("cupy")

    class _BadTensor:
        """桩张量：任何真实运算（此处为 sum）都抛运行期异常。"""

        def sum(self):
            raise RuntimeError("模拟 CUDA runtime 异常")

    stub.zeros = lambda *args, **kwargs: _BadTensor()  # 探活调用 zeros(1).sum() 即触发
    sys.modules["cupy"] = stub
    try:
        with _env(MINIMAL_GPU="1", CUPY_CACHE_DIR=None, CUDA_CACHE_PATH=None):
            be = importlib.reload(backend)
            use_gpu, np_mod = be.USE_GPU, be.np
    finally:
        if real_cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = real_cupy
    importlib.reload(backend)  # 复原为真实 cupy/numpy 状态
    assert use_gpu is False, "运行期探活失败时 USE_GPU 应回退为 False"
    assert np_mod is onp, "运行期运算异常时 np 应回退原生 numpy"
    print("runtime_probe_fallback PASS")


def test_runtime_probe_forces_device_sync():
    """探活必须把结果拷回 host 以强制与设备同步，否则异步 runtime 错误会被漏判。

    注入桩模块模拟「已装 cupy」：zeros(...) 与随后的 sum() 均正常返回一个驻留显存
    的 0 维张量（模拟 cupy 的异步语义——kernel 的 runtime 错误被延迟到同步点才抛出），
    而该张量仅在拷回 host（int()，即设备同步点）时才抛 RuntimeError。若探活丢弃
    sum() 返回值而不做 host 拷贝，就不会触发该异常、误判探活通过并把 np 指向坏掉的
    cupy，真实训练时才崩。此用例锁死「探活必须强制设备同步」这一行为。
    """
    real_cupy = sys.modules.get("cupy")
    stub = types.ModuleType("cupy")

    class _GpuTensor:
        """桩张量：sum() 返回驻留显存的结果，拷回 host（int()）时才暴露运行期故障。"""

        def sum(self):
            return _GpuTensor()  # 结果仍驻留设备，尚未同步

        def __int__(self):
            raise RuntimeError("模拟设备同步（host 拷贝）时的 CUDA runtime 异常")

    stub.zeros = lambda *args, **kwargs: _GpuTensor()  # 探活调用 zeros(1).sum()
    sys.modules["cupy"] = stub
    try:
        with _env(MINIMAL_GPU="1", CUPY_CACHE_DIR=None, CUDA_CACHE_PATH=None):
            be = importlib.reload(backend)
            use_gpu, np_mod = be.USE_GPU, be.np
    finally:
        if real_cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = real_cupy
    importlib.reload(backend)  # 复原为真实 cupy/numpy 状态
    assert use_gpu is False, "探活未强制设备同步时 USE_GPU 应回退为 False"
    assert np_mod is onp, "探活未强制设备同步时 np 应回退原生 numpy"
    print("runtime_probe_forces_device_sync PASS")


# ─────────────────────────── GPU 冒烟（子进程内执行）───────────────────────────
# 独立进程隔离显存与模块状态；以 MINIMAL_GPU=1 启动，不预设缓存目录环境变量，
# 由 backend 自身的 setdefault 完成重定向——编译缓存若不可写（沙箱拦截未解除），
# 首次真实 kernel 运算即报错、子进程非零退出，据此验证拦截已解除。
_GPU_SMOKE_CODE = r'''
import sys
import time

sys.path.insert(0, @@ROOT@@)

import numpy as onp

import model.backend as backend

if not backend.USE_GPU:
    print("SKIP")
    sys.exit(0)

from model.gpt import GPT
from train import AdamW

V, D, H, L, CTX = 64, 64, 4, 2, 32
onp.random.seed(0)
backend.np.random.seed(0)
model = GPT(vocab_size=V, d_model=D, n_head=H, n_layer=L, ctx_len=CTX)
opt = AdamW(model, lr=1e-3)
x = backend.np.asarray(onp.random.randint(0, V, size=(8, CTX)))
y = backend.np.asarray(onp.random.randint(0, V, size=(8, CTX)))

times = []
loss = None
for _ in range(20):
    t0 = time.perf_counter()
    logits = model.forward(x)
    loss = model.loss(logits, y)
    model.zero_grad()
    model.backward()
    opt.step()
    backend.np.cuda.runtime.deviceSynchronize()   # 同步，测真实单步耗时
    times.append((time.perf_counter() - t0) * 1000.0)

props = backend.np.cuda.runtime.getDeviceProperties(0)
_free_mem, total_mem = backend.np.cuda.runtime.memGetInfo()
print("RESULT", float(loss), min(times), sum(times) / len(times),
      total_mem, props["name"].decode())
'''


def test_gpu_smoke_train_20_steps():
    """小模型（d64/h4/L2/ctx32）在 GPU 上跑 20 步训练：loss 有限、无缓存写入报错。"""
    env = dict(os.environ)
    env["MINIMAL_GPU"] = "1"
    env.pop("CUPY_CACHE_DIR", None)     # 交给 backend 的 setdefault 重定向缓存
    env.pop("CUDA_CACHE_PATH", None)
    code = _GPU_SMOKE_CODE.replace("@@ROOT@@", repr(ROOT))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise AssertionError(f"GPU 冒烟子进程失败，退出码 {result.returncode}")
    if "SKIP" in result.stdout:
        print("跳过（GPU 不可用）")
        return
    line = next(l for l in result.stdout.splitlines() if l.startswith("RESULT"))
    parts = line.split(" ", 5)
    loss = float(parts[1])
    fast_ms, mean_ms = float(parts[2]), float(parts[3])
    total_mem, dev_name = int(parts[4]), parts[5]
    assert onp.isfinite(loss), f"20 步训练后 loss 非有限: {loss}"
    print(f"gpu_smoke_train_20_steps PASS  loss={loss:.4f}  "
          f"每步 {mean_ms:.2f} ms（最快 {fast_ms:.2f} ms）  "
          f"GPU={dev_name}  显存 {total_mem / 1024 ** 3:.2f} GiB")


if __name__ == "__main__":
    test_cache_dirs_redirected()
    test_setdefault_keeps_user_value()
    test_cpu_default_backend()
    test_gpu_enabled_backend()
    test_runtime_probe_fallback()
    test_runtime_probe_forces_device_sync()
    test_gpu_smoke_train_20_steps()
    print("\n全部 GPU 后端单测通过")
