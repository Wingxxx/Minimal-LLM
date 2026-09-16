"""计算后端总闸：默认 numpy（CPU）；设 MINIMAL_GPU=1 且 cupy 可用时切 GPU。

用法：所有模块统一 `from .backend import np`（train.py 在包外写绝对导入），
训练时设环境变量 MINIMAL_GPU=1 即全链路跑 GPU，不设则保持纯 numpy（CPU）。
设计目标：模型代码零改动，CPU 为默认教学路径，GPU 为可选加速拓展。

GPU 可用性判定分两步：cupy 可导入（安装层）+ 运行期探活（驱动/runtime 层），
任一不满足即回退纯 numpy，保证 GPU 为纯可选加速项、绝不影响 CPU 主路径。
"""
import os
import sys

# 原生 numpy（始终可用；npz 存档/数组搬运等场景需要它）
import numpy as onp

# 项目根目录：用于把 cupy/cuda 编译缓存重定向到项目内 _probe/ 下
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 编译缓存目录重定向：CUPY_CACHE_DIR 存 cupy 内核（cubin），CUDA_CACHE_PATH 存 CUDA JIT
# 编译产物。必须在 import cupy 之前设置——cupy 在导入期即读取这两个变量确定缓存落盘
# 位置，设晚了无效。setdefault 不覆盖调用方显式设置的值，仅在未设置时给出项目内默认值，
# 从而避开对用户主目录（如 ~/.cupy）的写入限制。
os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(_BASE, "_probe", "cupy_cache"))
os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(_BASE, "_probe", "cuda_cache"))

# cupy 为可选依赖：装了就绪，没装也不影响 CPU 路径
try:
    import cupy as _cupy
except ImportError:  # 未安装 cupy：GPU 后端不可用
    _cupy = None


def _cupy_runtime_ok():
    """运行期探活：cupy 可导入但 CUDA 驱动/runtime 异常时返回 False。

    仅捕获 ImportError 不足以覆盖「已装 cupy、但驱动缺失/版本不匹配/runtime 故障」
    的场景：此类故障在首次真实运算时才暴露，且抛出的是 CUDARuntimeError 等运行期
    异常而非 ImportError。这里触发一次真实 kernel 并强制与设备同步（把结果拷回
    host），捕获任意异常即判定 GPU 后端不可用，交由调用方回退纯 numpy。
    """
    if _cupy is None:
        return False
    try:
        # 触发真实 kernel 并把结果拷回 host（int 转换强制设备同步）；
        # 仅丢弃 sum() 返回值不会同步，kernel 的异步 runtime 错误将被漏判。
        int(_cupy.zeros(1).sum())
        return True
    except Exception as exc:  # 故意捕获全部异常：任何一种失败都意味着 GPU 不可用
        # 打印异常类型名，便于区分驱动/runtime 故障与实现自身的非预期异常（bug）
        print(f"[backend] cupy 已导入但运行期探活失败，回退 numpy(CPU)："
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return False


# 开关：环境变量 MINIMAL_GPU=1、cupy 可导入且运行期探活通过，才启用 GPU 后端。
# 未置 MINIMAL_GPU 时不调用探活（短路），CPU 路径零 GPU 交互。
USE_GPU = os.environ.get("MINIMAL_GPU") == "1" and _cupy_runtime_ok()

# 计算张量库：默认 numpy（CPU），启用时整库替换为 cupy（API 兼容）
np = _cupy if USE_GPU else onp


def as_numpy(a):
    """把张量转成原生 numpy 数组：GPU(cupy) 数组显式拷回 CPU（.get()），CPU 数组原样返回。

    注意不能用 onp.asarray(a) 走隐式转换——cupy 出于安全会直接报错拒绝，
    必须经 .get() 显式搬运。numpy 数组没有 .get()，走 onp.asarray 兜底。
    """
    return a.get() if hasattr(a, "get") else onp.asarray(a)
