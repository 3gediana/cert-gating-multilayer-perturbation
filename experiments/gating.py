"""门控规则：running z 分数 + 阈值告警 + hysteresis（pilot-design.md §2）。"""
import math

CHANNELS = ["drift", "bound", "beta"]


class RunningZ:
    """跨任务 running 均值/方差的 z 分数；样本 < 2 时返回 0（不告警）。"""

    def __init__(self):
        self.values = []

    def observe(self, v):
        self.values.append(float(v))

    def z(self, v):
        if len(self.values) < 2:
            return 0.0
        mu = sum(self.values) / len(self.values)
        var = sum((x - mu) ** 2 for x in self.values) / len(self.values)
        sd = math.sqrt(var) + 1e-8
        return (float(v) - mu) / sd


def decide(zs, active, prev_action, tau_hi=2.0, band=0.3):
    """active: 本臂进入告警的通道子集。
    返回 (action, max_z, override)；action in {'merge', 'isolate'}。
    hysteresis: 上一动作 isolate 且 max_z 落在 [tau-band, tau] 时维持 isolate。"""
    max_z = max(zs[c] for c in active) if active else 0.0
    alarm = max_z > tau_hi
    override = False
    if not alarm and prev_action == "isolate" and max_z > tau_hi - band:
        override = True
    action = "isolate" if (alarm or override) else "merge"
    return action, max_z, override
