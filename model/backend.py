"""计算后端总闸：默认 numpy（CPU）；设 MINIMAL_GPU=1 且已装 cupy 时切 GPU。

用法：所有模块统一 `from .backend import np`（train.py 在包外写绝对导入），
训练时设环境变量 MINIMAL_GPU=1 即全链路跑 GPU，不设则保持纯 numpy（CPU）。
设计目标：模型代码零改动，CPU 为默认教学路径，GPU 为可选加速拓展。
"""
import os

# 原生 numpy（始终可用；npz 存档/数组搬运等场景需要它）
import numpy as onp

# cupy 为可选依赖：装了就绪，没装也不影响 CPU 路径
try:
    import cupy as _cupy
except ImportError:  # 未安装 cupy：GPU 后端不可用
    _cupy = None

# 开关：环境变量 MINIMAL_GPU=1 且 cupy 可导入才启用 GPU
USE_GPU = os.environ.get("MINIMAL_GPU") == "1" and _cupy is not None

# 计算张量库：默认 numpy（CPU），启用时整库替换为 cupy（API 兼容）
np = _cupy if USE_GPU else onp


def as_numpy(a):
    """把张量转成原生 numpy 数组：GPU(cupy) 数组显式拷回 CPU（.get()），CPU 数组原样返回。

    注意不能用 onp.asarray(a) 走隐式转换——cupy 出于安全会直接报错拒绝，
    必须经 .get() 显式搬运。numpy 数组没有 .get()，走 onp.asarray 兜底。
    """
    return a.get() if hasattr(a, "get") else onp.asarray(a)
