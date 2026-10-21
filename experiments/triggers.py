"""三种触发量的可计算实现（pilot-design.md §3）。

- 表征层：线性 CKA 稳定度 -> 漂移 d = 1 - S
- 更新层：谱干涉界 B = ||dW||_2 * max_r ||J_r||_*（核范数）
- 几何层：锚泄漏 beta = max_r ||Q^T J_r^T||_F / ||J_r||_F
  （Q = span{最近两个试训练 dW} 的正交基）
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad as func_grad, vmap


def linear_cka(X, Y, eps=1e-8):
    """线性 CKA（Kornblith）：X, Y 为 (m, d) 表征矩阵。"""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    K = Xc.T @ Yc
    num = torch.linalg.matrix_norm(K) ** 2
    den = torch.linalg.matrix_norm(Xc.T @ Xc) * torch.linalg.matrix_norm(Yc.T @ Yc)
    return float((num / (den + eps)).clamp(0.0, 1.0))


def anchor_jacobians(model, params, tasks, assignments, device):
    """逐样本梯度矩阵 J_r（m x P）：锚任务 r 的损失对全部参数的梯度，
    routing 走任务 r 自己的 slot。params 为 requires_grad 的张量 dict。"""
    names = [n for n, _ in model.named_parameters()]
    Js = {}
    for r, task in enumerate(tasks):
        X = task["Xan"].to(device)
        Y = task["yan"].to(device)
        slot = assignments[r]

        def loss_fn(p, slot, head, x, y):
            logits = functional_call(model, p, (x, slot, head))
            return F.cross_entropy(logits, y, reduction="mean")

        per_sample = vmap(func_grad(loss_fn), in_dims=(None, None, None, 0, 0))(
            params, slot, r, X, Y
        )
        m = X.shape[0]
        Js[r] = torch.cat([per_sample[n].reshape(m, -1) for n in names], dim=1)
    return Js


def spectral_bound(dW, Js):
    """B = ||dW||_2 * max_r ||J_r||_*（核范数 = 奇异值之和）。"""
    dW_norm = torch.linalg.vector_norm(dW)
    worst = 0.0
    for J in Js.values():
        nuc = torch.linalg.svdvals(J).sum()
        worst = max(worst, float(nuc.detach()))
    return float(dW_norm) * worst


def subspace_leakage(deltas, Js):
    """beta = max_r ||Q^T J_r^T||_F / ||J_r||_F，Q = span(deltas) 的正交基。"""
    D = torch.stack(deltas, dim=1)  # (P, k)
    Q, _ = torch.linalg.qr(D)
    worst = 0.0
    for J in Js.values():
        proj = Q.T @ J.T  # (k, m)
        num = torch.linalg.matrix_norm(proj).detach()
        den = torch.linalg.matrix_norm(J).detach()
        worst = max(worst, float(num / (den + 1e-12)))
    return worst
