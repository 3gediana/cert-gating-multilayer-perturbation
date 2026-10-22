"""门控规则：running z 分数 + 阈值告警 + hysteresis（pilot-design.md §2）。

F2 修复候选 1（2026-10-23 experiment-redesign 单元；三候选预注册于
notes/DECISIONS.md 2026-10-22 条 + workflow.json experiment-redesign.strategy）：
- detrend：TrendAdjustedZ——先减任务序号线性项再算 running z。根因：bound 随任务
  序号单调增长（非平稳），RunningZ 全历史均值滞后 → 晚期持续越界、早期合并 → 边界
  振荡；去趋势直接移除非平稳源。
- persist：decide(..., persist=k)——动作持续期平滑，连续 k 个边界同向确认才切换
  动作（进入 isolate 需连续 k 次告警，退出需连续 k 次不告警）。
- taubump：提高 tau_hi（train.py --tau-hi）。按 workflow.json 约定，阈值类修法只作
  敏感性分析报告、防止事后调参，不选为正式规则。

F2 修复候选 2（2026-10-23 experiment-redesign-2 单元；候选④集预注册于
notes/DECISIONS.md 2026-10-23 条，跑前冻结）：
- asym：decide(..., k_in=2, k_out=3)——非对称持续期，进 isolate 需连续 k_in 次告警、
  退需连续 k_out=次不告警（放慢退出、进入保持 k_in=2 保灵敏度）。
- robust：RobustZ——中位数/MAD 稳健 z（对 ramp 末端/尖峰不敏感）。
- ewma：EWMAZ——指数加权 z；局部自适应均值=隐式去趋势（只对局部水平偏离告警，
  不做全局去趋势 → 不直接激活全部通道，针对 detrend 失败机制）。
- detrend-bound：仅 bound 通道用 TrendAdjustedZ（train.py per-channel 装配；
  drift/beta 保持 RunningZ——drift 原始 z 恒负，去趋势才转正越界）。
"""
import math

CHANNELS = ["drift", "bound", "beta"]


class RunningZ:
    """跨任务 running 均值/方差的 z 分数；样本 < 2 时返回 0（不告警）。
    接口与 TrendAdjustedZ 对齐：z(i, v)/observe(i, v)，i（任务序号）被忽略。"""

    def __init__(self):
        self.values = []

    def observe(self, i, v):
        self.values.append(float(v))

    def z(self, i, v):
        if len(self.values) < 2:
            return 0.0
        mu = sum(self.values) / len(self.values)
        var = sum((x - mu) ** 2 for x in self.values) / len(self.values)
        sd = math.sqrt(var) + 1e-8
        return (float(v) - mu) / sd


class TrendAdjustedZ:
    """去趋势 running z：对历史样本最小二乘拟合 raw ~ a*i + b（i = 任务序号），
    当前值先减趋势值，再对残差序列的 running 均值/标准差算 z。
    样本 < 3 时返回 0（<3 点线性拟合无残差自由度，与"第 3 任务起门控"对齐）。"""

    def __init__(self):
        self.idx = []
        self.values = []

    def observe(self, i, v):
        self.idx.append(float(i))
        self.values.append(float(v))

    def _fit(self):
        n = len(self.values)
        mx = sum(self.idx) / n
        my = sum(self.values) / n
        sxx = sum((x - mx) ** 2 for x in self.idx)
        if sxx <= 0:
            return None
        a = sum((x - mx) * (y - my) for x, y in zip(self.idx, self.values)) / sxx
        b = my - a * mx
        resid = [y - (a * x + b) for x, y in zip(self.idx, self.values)]
        return a, b, resid

    def z(self, i, v):
        if len(self.values) < 3:
            return 0.0
        fit = self._fit()
        if fit is None:
            return 0.0
        a, b, resid = fit
        mr = sum(resid) / len(resid)
        sd = math.sqrt(sum((r - mr) ** 2 for r in resid) / len(resid)) + 1e-8
        return (float(v) - (a * float(i) + b) - mr) / sd


class RobustZ:
    """稳健 running z：中位数/MAD（样本 <3 返回 0，与 TrendAdjustedZ 对齐）。
    MAD=0（历史样本 >50% 相同）时用相对中位数的地板 sd=1e-8*(|med|+1e-12)
    防除零——恒定历史后的跳变视为异常（大幅偏离），恒定量级内的噪声不告警。"""

    def __init__(self):
        self.values = []

    def observe(self, i, v):
        self.values.append(float(v))

    @staticmethod
    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def z(self, i, v):
        if len(self.values) < 3:
            return 0.0
        med = self._median(self.values)
        mad = self._median([abs(x - med) for x in self.values])
        sd = 1.4826 * mad
        floor = 1e-8 * (abs(med) + 1e-12)
        if sd < floor:
            sd = floor
        return (float(v) - med) / sd


class EWMAZ:
    """EWMA z：指数加权均值/方差（alpha 默认 0.3，样本 <3 返回 0）。
    局部自适应均值跟踪持续趋势（隐式去趋势）——只有相对局部水平的偏离产生 z；
    不做全局去趋势，因此不直接改变其他通道的告警结构（针对 detrend 的失败机制）。
    方差用偏差修正分母 1-(1-alpha)^{2n}；m2=0（恒定历史）时加 1e-8 地板。"""

    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.mean = None
        self.m2 = 0.0
        self.n = 0

    def observe(self, i, v):
        v = float(v)
        if self.n == 0:
            self.mean = v
        else:
            diff = v - self.mean
            self.mean += self.alpha * diff
            self.m2 = (1 - self.alpha) * (self.m2 + self.alpha * diff * diff)
        self.n += 1

    def z(self, i, v):
        if self.n < 3:
            return 0.0
        var = self.m2 / (1 - (1 - self.alpha) ** (2 * self.n))
        sd = math.sqrt(max(var, 0.0)) + 1e-8
        return (float(v) - self.mean) / sd


def decide(zs, active, prev_action, tau_hi=2.0, band=0.3, persist=1,
           k_in=None, k_out=None, state=None):
    """active: 本臂进入告警的通道子集。
    返回 (action, max_z, override, state)；action in {'merge', 'isolate'}。
    hysteresis: 上一动作 isolate 且 max_z 落在 [tau-band, tau] 时维持 isolate。
    persist=k>1: 动作持续期平滑——连续 k 个边界同向确认才切换动作；hysteresis
    override 仍优先（只可能延长 isolate，不制造新 switch）。
    k_in/k_out 非 None（非对称持续期）：进 isolate 需连续 k_in 个边界告警、
    退需连续 k_out 个边界不告警（persist 被忽略）；override 逻辑不变。"""
    max_z = max(zs[c] for c in active) if active else 0.0
    alarm = max_z > tau_hi
    st = dict(state) if state else {"alarm": 0, "calm": 0}
    st["alarm"] = st.get("alarm", 0) + 1 if alarm else 0
    st["calm"] = st.get("calm", 0) + 1 if not alarm else 0
    override = False
    if k_in is not None:
        k_out_eff = k_out if k_out is not None else k_in
        want = "isolate" if alarm else "merge"
        if want == prev_action:
            action = want
        else:
            streak = st["alarm"] if want == "isolate" else st["calm"]
            need = k_in if want == "isolate" else k_out_eff
            action = want if streak >= need else prev_action
    elif persist <= 1:
        action = "isolate" if alarm else "merge"
    else:
        want = "isolate" if alarm else "merge"
        if want == prev_action:
            action = want
        else:
            streak = st["alarm"] if want == "isolate" else st["calm"]
            action = want if streak >= persist else prev_action
    if action == "merge" and prev_action == "isolate" and max_z > tau_hi - band:
        override = True
        action = "isolate"
    return action, max_z, override, st
