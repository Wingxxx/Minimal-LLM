"""GPT 端到端：loss 有限、KV Cache 一致性、保存加载一致性、参数量"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT

def test_gpt_loss_and_backward():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    x = np.random.randint(0, 30, size=(4, 10))
    y = np.random.randint(0, 30, size=(4, 10))
    logits = gpt.forward(x)
    loss = gpt.loss(logits, y)
    assert logits.shape == (4, 10, 30), logits.shape
    assert np.isfinite(loss), f"loss 非有限: {loss}"
    assert 2.0 < loss < 5.0, f"loss 异常: {loss}"
    gpt.zero_grad(); gpt.backward()
    assert np.all(np.isfinite(gpt.d_tok_emb)), "tok_emb 梯度非有限"
    print("GPT loss/backward PASS", round(float(loss), 4))

def test_gpt_kv_cache():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    x = np.random.randint(0, 30, size=(1, 6))
    y_full = gpt.forward(x)
    gpt2 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2.load_params(gpt.dump_params())
    head_dim = gpt2.d_model // gpt2.n_head
    cache = [[np.zeros((1, gpt2.n_head, 0, head_dim)),
              np.zeros((1, gpt2.n_head, 0, head_dim))] for _ in range(gpt2.n_layer)]
    first = gpt2.forward(x[:, :1], cache)
    rest = [gpt2.forward(x[:, t:t+1], cache) for t in range(1, 6)]
    y_step = np.concatenate([first] + rest, axis=1)
    err = np.abs(y_full - y_step).max()
    assert err < 1e-8, f"GPT KV Cache 不一致 err={err}"
    print("GPT KV Cache PASS", err)

def test_save_load():
    gpt1 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2.load_params(gpt1.dump_params())
    x = np.random.randint(0, 30, size=(2, 8))
    assert np.abs(gpt1.forward(x) - gpt2.forward(x)).max() < 1e-10
    print("save/load consistency PASS")

def test_parameter_count():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    n = sum(p.size for p in gpt.dump_params().values())
    print("param count:", n)
    assert n > 0

if __name__ == "__main__":
    test_gpt_loss_and_backward(); test_gpt_kv_cache(); test_save_load(); test_parameter_count()
    print("ALL GPT TESTS PASS")
