"""Slots 架构：K 个 trunk 槽位 + per-task 线性头。

trunk0 = 共享并入目标（merge 训练只动它）；trunk1..K-1 = 隔离池（isolate 任务独占）。
五臂总参数量由构造相同（同样的 trunk 数/宽度 + T 个头）。
"""
import torch
import torch.nn as nn


class Trunk(nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.fc1 = nn.Linear(d, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class SlotsModel(nn.Module):
    def __init__(self, d, width, k, T, slots):
        super().__init__()
        self.trunks = nn.ModuleList([Trunk(d, width) for _ in range(slots)])
        self.heads = nn.ModuleList([nn.Linear(width, k) for _ in range(T)])

    def forward(self, x, slot, head):
        return self.heads[head](self.trunks[slot](x))

    def train_params(self, slot, head):
        """只训练指定 trunk 槽位 + 指定任务的 head（隔离训练不污染 trunk0）。"""
        ps = []
        for n, p in self.named_parameters():
            if n.startswith(f"trunks.{slot}.") or n.startswith(f"heads.{head}."):
                ps.append(p)
        return ps

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
