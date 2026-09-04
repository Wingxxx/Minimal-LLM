from .backend import np  # 计算后端：默认 numpy(CPU)，MINIMAL_GPU=1 切 cupy(GPU)


class Linear:
    """线性层：y = x @ W + b。

    前向将输入投影到输出维度；反向按链式法则计算梯度并回传。
    """

    def __init__(self, in_f, out_f):
        # W 初始化公式：W[i,j] ~ randn(in_f, out_f) / sqrt(in_f)，使每个输出的输入方差 ≈ 1
        self.W = np.random.randn(in_f, out_f) / np.sqrt(in_f)
        # b 初始化：全 0
        self.b = np.zeros(out_f)
        # 梯度缓存：dW 与 W、db 与 b 形状一致（形状铁律）
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # 缓存前向输入 x，backward 中计算 dW = x^T @ dy 时需要
        self.x = None

    def forward(self, x):
        # 缓存输入 x
        self.x = x
        # 前向公式：y = x @ W + b，形状 [B,T,d] @ [d,out] + [out] → [B,T,out]
        return x @ self.W + self.b

    def backward(self, dy):
        # 将 x、dy 展平为 [B*T, d] 与 [B*T, out]，一次矩阵乘法即可累加所有位置
        flat_x = self.x.reshape(-1, self.x.shape[-1])
        flat_dy = dy.reshape(-1, dy.shape[-1])
        # 反向公式：dW = x^T @ dy（权重共享，跨所有位置累积），形状 [d, out]
        self.dW += flat_x.T @ flat_dy
        # 反向公式：db = Σ dy（沿 batch/T 维求和），形状 [out]
        self.db += dy.sum(axis=tuple(range(dy.ndim - 1)))
        # 反向公式：dx = dy @ W^T，回传至上一层，形状 [B,T,d]
        return dy @ self.W.T

    def get_params(self):
        # 返回 (名称, 参数) 列表，供优化器遍历更新
        return [('W', self.W), ('b', self.b)]

    def get_grad(self, name):
        # 按参数名返回梯度，供优化器读取
        if name == 'W':
            return self.dW
        if name == 'b':
            return self.db
        return None

    def set_param(self, name, val):
        # 按参数名设置参数值（加载模型权重时使用）
        if name == 'W':
            self.W = val
        elif name == 'b':
            self.b = val

    def zero_grad(self):
        # 清零梯度缓存（backward 使用 += 累加，每训练步前必须清零）
        self.dW[:] = 0
        self.db[:] = 0


class LayerNorm:
    """层归一化：对每个位置的特征做标准化，再乘 γ 加 β。

    前向把每个位置的特征分布拉回均值 0、方差 1；反向按链式法则回传梯度。
    """

    def __init__(self, dim, eps=1e-5):
        # dim：特征维度 d；eps：防除零小常数
        self.dim, self.eps = dim, eps
        # 可学习参数初始化（对应公式 y = x̂·γ + β）：γ=1、β=0，即先"只归一化、不改写"
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        # 梯度缓存：dgamma 与 gamma、dbeta 与 beta 形状一致（形状铁律）
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        # 前向中间结果缓存（x/μ/σ²/x̂），反向计算梯度时需要
        self.x = self.mean = self.var = self.xhat = None

    def forward(self, x):
        # 缓存输入 x
        self.x = x
        # 公式：μ = mean(x)、σ² = var(x)，沿特征维 (axis=-1)；keepdims 保留 [B,T,1] 便于与 x 广播
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        # 公式：x̂ = (x − μ) / √(σ² + ε)，归一化使每个位置均值 0、方差 1
        self.xhat = (x - self.mean) / np.sqrt(self.var + self.eps)
        # 公式：y = x̂·γ + β，乘可学习缩放、加可学习偏移后返回
        return self.xhat * self.gamma + self.beta

    def backward(self, dy):
        # 公式：dx̂ = dy · γ（x̂ 到 y 的系数是 γ），逐元素相乘，形状 [B,T,d]
        dxhat = dy * self.gamma
        # 公式：dγ = Σ(dy · x̂)，沿 batch/T 维求和（γ 被所有位置共享，梯度累计），形状 [d]
        self.dgamma += (dy * self.xhat).sum(axis=tuple(range(dy.ndim - 1)))
        # 公式：dβ = Σ dy，沿 batch/T 维求和，形状 [d]
        self.dbeta += dy.sum(axis=tuple(range(dy.ndim - 1)))
        # 公式：dx = (dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂)) / √(σ²+ε)
        # 三项含义：直接路 − μ 的拉扯 − σ 的拉扯；mean 沿特征维 (axis=-1)，keepdims 便于广播
        d = dxhat - dxhat.mean(-1, keepdims=True) - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)
        # 公式：dx = d / √(σ² + ε)，回传至上一层，形状 [B,T,d]
        return d / np.sqrt(self.var + self.eps)

    def get_params(self):
        # 返回 (名称, 参数) 列表，供优化器遍历更新
        return [('gamma', self.gamma), ('beta', self.beta)]

    def get_grad(self, name):
        # 按参数名返回梯度，供优化器读取
        if name == 'gamma':
            return self.dgamma
        if name == 'beta':
            return self.dbeta
        return None

    def set_param(self, name, val):
        # 按参数名设置参数值（加载模型权重时使用）
        if name == 'gamma':
            self.gamma = val
        elif name == 'beta':
            self.beta = val

    def zero_grad(self):
        # 清零梯度缓存（backward 使用 += 累加，每训练步前必须清零）
        self.dgamma[:] = 0
        self.dbeta[:] = 0


class GELU:
    """高斯误差线性单元：y = 0.5·x·(1 + tanh(g))，给网络提供非线性。

    用 tanh 近似标准正态累积分布 Φ(x)；前向形状不变，反向按链式法则求导。
    """

    def __init__(self):
        # 前向中间结果缓存（x/g/tanh(g)），反向计算梯度时需要
        self.x = self.g = self.tanh_g = None

    def forward(self, x):
        # 缓存输入 x
        self.x = x
        # 公式：g = √(2/π)·(x + 0.044715·x³)，tanh 近似的内部参数
        self.g = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        # tanh(g)：S 形门控值，逼近标准正态累积分布 Φ(x)，范围 (-1, 1)
        self.tanh_g = np.tanh(self.g)
        # 公式：y = 0.5·x·(1 + tanh(g))，输出形状与输入一致 [B,T,d]
        return 0.5 * x * (1 + self.tanh_g)

    def backward(self, dy):
        # 公式：g'(x) = √(2/π)·(1 + 3·0.044715·x²)，g 对 x 的导数
        gprime = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x ** 2)
        # 公式：dx = dy · [0.5·(1 + tanh(g)) + 0.5·x·(1 − tanh²(g))·g'(x)]
        # 第一项来自 x 的直接作用，第二项来自 x 通过 g 的作用；形状与输入一致 [B,T,d]
        return dy * (0.5 * (1 + self.tanh_g) + 0.5 * self.x * (1 - self.tanh_g ** 2) * gprime)

    def get_params(self):
        # GELU 无可学习参数，返回空列表
        return []

    def get_grad(self, name):
        # 无可学习参数，恒返回 None
        return None

    def set_param(self, name, val):
        # 无可学习参数，空操作
        pass

    def zero_grad(self):
        # 无可学习参数，空操作
        pass

