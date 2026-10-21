"""合成持续学习任务序列 + Split-CIFAR10（phase 2）。

合成协议（pilot-design.md §4）：T 个任务，每任务 k 类；任务 t 的类中心 = 基类中心
经随机正交旋转 × 漂移缩放 (1+rho*t/(T-1))，各向同性噪声 noise。旋转制造表征冲突，
漂移制造可塑性压力。
"""
import numpy as np
import torch


def _rand_orthogonal(rng, d):
    Q, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(d, d)))
    return Q


def make_task_sequence(seed, T, d, k, train_n, test_n, anchor_n, drift=0.3, noise=0.5):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(k, d))
    tasks = []
    for t in range(T):
        R = _rand_orthogonal(rng, d)
        scale = 1.0 + drift * (t / max(T - 1, 1))
        centers = scale * (base @ R.T)

        def sample(n):
            y = rng.integers(0, k, size=n)
            X = centers[y] + noise * rng.normal(0.0, 1.0, size=(n, d))
            return X, y

        Xtr, ytr = sample(train_n)
        Xte, yte = sample(test_n)
        m = min(anchor_n, train_n)
        tasks.append({
            "Xtr": torch.tensor(Xtr, dtype=torch.float32),
            "ytr": torch.tensor(ytr, dtype=torch.long),
            "Xte": torch.tensor(Xte, dtype=torch.float32),
            "yte": torch.tensor(yte, dtype=torch.long),
            "Xan": torch.tensor(Xtr[:m], dtype=torch.float32),
            "yan": torch.tensor(ytr[:m], dtype=torch.long),
        })
    return tasks


def make_split_cifar(seed, T=5, classes_per_task=2, train_n=2000, test_n=500,
                     anchor_n=64, root="data/cifar10"):
    """Split-CIFAR10：T 个任务 × classes_per_task 类，输入展平为 3072 维 /255。
    数据下载到 root（.gitignore 已排除 data/）。"""
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(root=root, train=True, download=True)
    te = CIFAR10(root=root, train=False, download=True)

    def to_xy(ds):
        X = np.asarray(ds.data, dtype=np.float32).reshape(len(ds.data), -1) / 255.0
        y = np.asarray(ds.targets, dtype=np.int64)
        return X, y

    Xtr, ytr = to_xy(tr)
    Xte, yte = to_xy(te)
    rng = np.random.default_rng(seed)
    order = rng.permutation(10)
    tasks = []
    for t in range(T):
        cls = order[t * classes_per_task:(t + 1) * classes_per_task]
        remap = {int(c): j for j, c in enumerate(cls)}

        def pick(X, y, n):
            idx = np.concatenate([np.where(y == c)[0][:n] for c in cls])
            Xs, ys = X[idx], y[idx]
            ys = np.array([remap[int(v)] for v in ys], dtype=np.int64)
            perm = rng.permutation(len(ys))
            return Xs[perm], ys[perm]

        Xa, ya = pick(Xtr, ytr, train_n)
        Xb, yb = pick(Xte, yte, test_n)
        m = min(anchor_n, len(Xa))
        tasks.append({
            "Xtr": torch.tensor(Xa, dtype=torch.float32),
            "ytr": torch.tensor(ya, dtype=torch.long),
            "Xte": torch.tensor(Xb, dtype=torch.float32),
            "yte": torch.tensor(yb, dtype=torch.long),
            "Xan": torch.tensor(Xa[:m], dtype=torch.float32),
            "yan": torch.tensor(ya[:m], dtype=torch.long),
        })
    return tasks
