"""F2 修复聚合分析（experiment-redesign 阶段）：规则网格 × 5 种子 × 5 臂。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_f2fix.py \\
    --in results/pilot-f2fix-metrics.json \\
    --md results/pilot-f2fix-summary.md \\
    --json results/pilot-f2fix-summary.json

预注册 gate（workflow.json gate_step + pilot-design.md §7 F2）：
- switch<0.3：任一臂 action-switch rate。判读口径与 analyze_pilot.py 的 F2 实现
  一致——per-seed rate 的跨种子 max > 0.3 即触发，故验收主口径用 switch_max<0.3，
  同时报告 switch_mean<0.3（次口径，透明备查）。
- F̄ 不劣化于 0.0331（归档 pilot spectral/triplet 基准）：F_mean<=0.0331；
  另做同批配对（vs 原规则对照 none/p1/tau2，每种子 ΔF<=0.005 视为不劣化）。
- F1 退化检查按 pilot-design.md §8 判读细则：跨种子汇总 Spearman（原始+去趋势），
  单种子 n 小会假高，只作参考列。

规则标签：{f2_fix}/p{persist}/tau{tau_hi}。persist 轴必须进标签——f2_fix=persist
只是别名（行为由 --persist 决定），none/p2 与 persist/p2 逐位相同（复现性复证），
分析只取唯一单元，别名单元作确定性证据记录。
对照 = none/p1/tau2（原规则，Phase 1 触发 F2 的规则，同批重跑）。
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = ["baseline", "cka", "spectral", "beta", "triplet"]
CH = ["drift", "bound", "beta"]
CONTROL_KEY = ("none", 1, 2.0)   # 原规则（hysteresis 已开、无修复）
F_BENCH = 0.0331                 # 归档 pilot spectral/triplet F̄（gate_step 基准）
SW_THRESH = 0.3
PAIR_TOL = 0.005
GATE_ARMS = ["spectral", "triplet"]  # F̄ 基准臂（workflow gate_step 明示）


def rule_key(cfg):
    return (cfg["f2_fix"], int(cfg["persist"]), float(cfg["tau_hi"]))


def rule_label(key):
    return f"{key[0]}/p{key[1]}/tau{key[2]:g}"


def load_cells(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = {}
    for r in data["runs"]:
        key = rule_key(r["config"])
        cells.setdefault(key, {})[r["config"]["seed"]] = r
    # 别名单元：与另一单元行为逐位相同（f2_fix=persist 是 none+persist 的别名）。
    # 签名只取行为字段（F/动作/相关/参数），剔除 timings——计时逐 run 自然波动。
    BEHAVIOR = ["mean_forgetting", "per_task_forgetting", "mean_final_acc",
                "n_merge", "n_isolate", "n_forced", "action_switch_rate",
                "correlations", "param_count", "slots_used"]
    aliases, seen = {}, {}
    for key in sorted(cells):
        sig = json.dumps(
            [{a["arm"]: {k: a["summary"][k] for k in BEHAVIOR}
              for a in cells[key][s]["arms"]}
             for s in sorted(cells[key])], sort_keys=True)
        if sig in seen:
            aliases[rule_label(key)] = rule_label(seen[sig])
        else:
            seen[sig] = key
    return cells, aliases


def arm_stats(cell):
    """rule -> arm -> 每种子 F/switch + 汇总。"""
    out = {}
    for arm in ARMS:
        rows = []
        for seed in sorted(cell):
            s = next(x for x in cell[seed]["arms"] if x["arm"] == arm)["summary"]
            rows.append({"seed": seed, "F": s["mean_forgetting"],
                         "switch": s["action_switch_rate"],
                         "n_isolate": s["n_isolate"], "n_forced": s["n_forced"],
                         "acc": s["mean_final_acc"], "cost": s["trigger_cost_ratio"]})
        F = np.array([r["F"] for r in rows])
        sw = np.array([r["switch"] for r in rows])
        out[arm] = {
            "per_seed": rows,
            "F_mean": round(float(F.mean()), 4),
            "F_sd": round(float(F.std(ddof=1)), 4),
            "F_per_seed": [r["F"] for r in rows],
            "switch_per_seed": [r["switch"] for r in rows],
            "switch_mean": round(float(sw.mean()), 4),
            "switch_max": round(float(sw.max()), 4),
            "n_isolate_total": int(sum(r["n_isolate"] for r in rows)),
            "n_forced_total": int(sum(r["n_forced"] for r in rows)),
            "acc_mean": round(float(np.mean([r["acc"] for r in rows])), 4),
            "cost_mean": round(float(np.mean([r["cost"] for r in rows])), 4),
        }
    return out


def f1_pooled(cell):
    """F1 退化检查（预注册口径）：每臂跨种子汇总 Spearman（原始/去趋势）。
    单种子 |rho| 只作参考（n=9 会趋势性假高，pilot-design.md §8 判读细则）。"""
    out = {}
    for arm in ARMS:
        entry = {}
        for pair in combinations(CH, 2):
            key = f"{pair[0]}~{pair[1]}"
            pooled_x, pooled_y, per_seed = [], [], []
            for seed in sorted(cell):
                a = next(x for x in cell[seed]["arms"] if x["arm"] == arm)
                xa = [rec["raws"][pair[0]] for rec in a["records"][1:]]
                xb = [rec["raws"][pair[1]] for rec in a["records"][1:]]
                if len(set(xa)) <= 1 or len(set(xb)) <= 1:
                    continue
                per_seed.append(round(float(spearmanr(xa, xb).correlation), 4))
                pooled_x.append(np.asarray(xa, float))
                pooled_y.append(np.asarray(xb, float))
            if not pooled_x:
                entry[key] = {"note": "constant"}
                continue
            X = np.concatenate(pooled_x)
            Y = np.concatenate(pooled_y)
            pooled = float(spearmanr(X, Y).correlation)

            def detr(arrs):
                outs = []
                for v in arrs:
                    x = np.arange(1, len(v) + 1, dtype=float)
                    if len(v) > 2 and v.std() > 0:
                        b, a0 = np.polyfit(x, v, 1)
                        outs.append(v - (a0 + b * x))
                return np.concatenate(outs)

            d = float(spearmanr(detr(pooled_x), detr(pooled_y)).correlation)
            entry[key] = {
                "pooled_rho": round(pooled, 4),
                "detrended_rho": round(d, 4),
                "per_seed_rho": per_seed,
                "per_seed_max_abs": round(float(np.max(np.abs(per_seed))), 4),
                "f1_raw": bool(abs(pooled) > 0.9),
                "f1_detrended": bool(abs(d) > 0.9),
            }
        out[arm] = entry
    return out


def channel_firing(cell, tau_hi):
    """每臂每通道在决策点（i>=2）的越界次数（z>tau，从 records.zs 重算）。"""
    out = {}
    for arm in ARMS:
        counts = {c: 0 for c in CH}
        n_dec = 0
        for seed in sorted(cell):
            a = next(x for x in cell[seed]["arms"] if x["arm"] == arm)
            for rec in a["records"]:
                if rec["task"] >= 2:
                    n_dec += 1
                    for c in CH:
                        if rec["zs"][c] > tau_hi:
                            counts[c] += 1
        out[arm] = {"n_decision_points": n_dec, **counts}
    return out


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/pilot-f2fix-metrics.json")
    ap.add_argument("--md", default="results/pilot-f2fix-summary.md")
    ap.add_argument("--json", default="results/pilot-f2fix-summary.json")
    args = ap.parse_args()

    cells, aliases = load_cells(args.inp)
    control = cells[CONTROL_KEY]
    ctrl_stats = arm_stats(control)

    tab, f1, fire, verdicts = {}, {}, {}, {}
    for key in sorted(cells):
        lab = rule_label(key)
        if lab in aliases:
            continue  # 别名单元不进分析（与主单元逐位相同）
        tab[lab] = arm_stats(cells[key])
        f1[lab] = f1_pooled(cells[key])
        fire[lab] = channel_firing(cells[key], key[2])
        if key == CONTROL_KEY:
            continue
        detail, ok_all = {}, True
        for arm in ARMS:
            t = tab[lab][arm]
            sw_max_ok = t["switch_max"] < SW_THRESH + 1e-9
            sw_mean_ok = t["switch_mean"] < SW_THRESH + 1e-9
            d = {"switch_mean": t["switch_mean"], "switch_max": t["switch_max"],
                 "switch_max_ok": bool(sw_max_ok), "switch_mean_ok": bool(sw_mean_ok),
                 "n_isolate_total": t["n_isolate_total"]}
            if arm in GATE_ARMS:
                c = ctrl_stats[arm]
                dF = [round(f - b, 4) for f, b in zip(t["F_per_seed"], c["F_per_seed"])]
                lo, hi = bootstrap_ci(dF)
                d.update({
                    "F_mean": t["F_mean"], "F_per_seed": t["F_per_seed"],
                    "F_le_bench": bool(t["F_mean"] <= F_BENCH + 1e-9),
                    "dF_vs_control_per_seed": dF,
                    "dF_mean": round(float(np.mean(dF)), 4),
                    "dF_ci95": [lo, hi],
                    "paired_not_worse": bool(all(x <= PAIR_TOL for x in dF)),
                })
                ok_all = ok_all and sw_max_ok and d["F_le_bench"] and d["paired_not_worse"]
            else:
                ok_all = ok_all and sw_max_ok
            detail[arm] = d
        verdicts[lab] = {
            "pass_primary_max_reading": bool(ok_all),
            "pass_secondary_mean_reading": bool(all(
                tab[lab][a]["switch_mean"] < SW_THRESH + 1e-9 for a in ARMS)
                and all(tab[lab][a]["F_mean"] <= F_BENCH + 1e-9
                        and all(x <= PAIR_TOL for x in
                                [round(f - b, 4) for f, b in
                                 zip(tab[lab][a]["F_per_seed"], ctrl_stats[a]["F_per_seed"])])
                        for a in GATE_ARMS)),
            "detail": detail,
        }

    f1_max = {}
    for lab, arms_d in f1.items():
        mx = 0.0
        for entry in arms_d.values():
            for e in entry.values():
                if "pooled_rho" in e:
                    mx = max(mx, abs(e["pooled_rho"]))
        f1_max[lab] = round(mx, 4)

    rules = sorted(tab)
    payload = {
        "analysis": "F2 修复验收（experiment-redesign）：预注册三候选（detrend 去趋势 z / "
                    "tau 提高 / persist 动作持续期平滑 k）× pilot T=10 width64 × 5 种子 × 5 臂。"
                    "验收=switch<0.3（主口径 per-seed max，与 analyze_pilot.py F2 实现一致；"
                    "次口径 mean）且 F̄ 不劣化于 0.0331 + 同批配对不劣化（vs 原规则 none/p1/tau2）。",
        "control": rule_label(CONTROL_KEY),
        "benchmark_F": F_BENCH,
        "switch_threshold": SW_THRESH,
        "paired_tolerance": PAIR_TOL,
        "alias_cells": aliases,
        "grid": {rule_label(k): sorted(cells[k]) for k in sorted(cells)},
        "arm_table": tab,
        "control_table": ctrl_stats,
        "f1_pooled_cross_seed": f1,
        "f1_max_abs_pooled_per_rule": f1_max,
        "channel_firing": fire,
        "verdicts": verdicts,
    }
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                               encoding="utf-8")

    # ---- markdown
    L = []
    L.append("# F2 修复实验：门控规则重设计（experiment-redesign 阶段）\n")
    L.append("> 预注册三候选（pilot-design.md §7 F2 处置 + workflow.json gate_step）："
             "①detrend 去趋势 z ②提高 tau（阈值类，按约定仅敏感性分析）"
             "③persist 动作持续期平滑（连续 k 个边界同向确认才切换）。"
             "**验收：switch<0.3 且 F̄ 不劣化于 0.0331**（归档 pilot spectral/triplet 基准）。\n")
    L.append(f"> 对照 = `{rule_label(CONTROL_KEY)}`（Phase 1 触发 F2 的原规则，同批重跑："
             f"spectral switch {ctrl_stats['spectral']['switch_mean']}/"
             f"{ctrl_stats['spectral']['switch_max']}、F̄ {ctrl_stats['spectral']['F_mean']}"
             "——复现 Phase 1 触发读数，对照有效）。"
             "switch 主口径 = per-seed rate 跨种子 max（与 analyze_pilot.py 的 F2 判读一致），"
             "次口径 = mean，双口径列报。\n")
    L.append("## 网格（唯一单元 × 5 种子；别名单元见 JSON）\n")
    L.append("| 规则 | 说明 | seeds |")
    L.append("|---|---|---|")
    desc = {
        "none/p1/tau2": "原规则（对照；hysteresis 已开）",
        "none/p2/tau2": "候选③ persist k=2",
        "none/p3/tau2": "候选③ persist k=3",
        "none/p1/tau3": "候选② tau=3（敏感性）",
        "none/p2/tau3": "候选②+③",
        "detrend/p1/tau2": "候选① 去趋势 z",
        "detrend/p2/tau2": "候选①+③",
    }
    for lab in rules:
        L.append(f"| {lab} | {desc.get(lab, '')} | {len(tab[lab]['baseline']['per_seed'])} |")
    if aliases:
        L.append("\n别名单元（逐位复现主单元，作确定性证据，不进分析）："
                 + "；".join(f"{k}≡{v}" for k, v in aliases.items()))
    L.append("\n## 主表（5 种子；F̄=mean±sd；switch=mean/max（每种子）；isolate 总数）\n")
    L.append("| 规则 | 臂 | F̄±sd | F̄/种子 | switch mean/max | isolate(forced) | acc | cost |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab in rules:
        for arm in ARMS:
            t = tab[lab][arm]
            fps = "/".join(f"{v:.4f}" for v in t["F_per_seed"])
            sws = "/".join(f"{v:.2f}" for v in t["switch_per_seed"])
            star = " **(对照)**" if lab == rule_label(CONTROL_KEY) and arm == "spectral" else ""
            L.append(f"| {lab}{star} | {arm} | {t['F_mean']:.4f}±{t['F_sd']:.4f} | {fps} | "
                     f"{t['switch_mean']:.2f}/{t['switch_max']:.2f} ({sws}) | "
                     f"{t['n_isolate_total']}({t['n_forced_total']}) | {t['acc_mean']:.4f} | "
                     f"{t['cost_mean']:.3f} |")
    L.append("\n## 验收判读（主口径：switch_max<0.3 且 F̄≤0.0331 且配对不劣化）\n")
    L.append("| 规则 | spectral | triplet | 其他臂 switch_max | 总判(主/次) |")
    L.append("|---|---|---|---|---|")

    def cell(d):
        if "F_mean" not in d:
            return f"sw={'✓' if d['switch_max_ok'] else '✗'} ({d['switch_mean']:.2f}/{d['switch_max']:.2f})"
        return (f"sw={'✓' if d['switch_max_ok'] else '✗'}({d['switch_mean']:.2f}/{d['switch_max']:.2f}) "
                f"F={'✓' if d['F_le_bench'] else '✗'}({d['F_mean']:.4f}) "
                f"pair={'✓' if d['paired_not_worse'] else '✗'}(Δ={d['dF_mean']:+.4f})")

    for lab, v in verdicts.items():
        sp, tp = v["detail"]["spectral"], v["detail"]["triplet"]
        others = {a: v["detail"][a]["switch_max"] for a in ["baseline", "cka", "beta"]}
        oth = " ".join(f"{a}:{x:.2f}{'✓' if x < SW_THRESH else '✗'}" for a, x in others.items())
        L.append(f"| {lab} | {cell(sp)} | {cell(tp)} | {oth} | "
                 f"{'**PASS**' if v['pass_primary_max_reading'] else 'FAIL'}/"
                 f"{'PASS' if v['pass_secondary_mean_reading'] else 'FAIL'} |")
    L.append("\n## 配对差（规则 − 对照 none/p1/tau2，每种子 F̄，spectral/triplet）\n")
    for lab, v in verdicts.items():
        for arm in GATE_ARMS:
            d = v["detail"][arm]
            L.append(f"- {lab} {arm}: ΔF̄={d['dF_mean']:+.4f} "
                     f"CI95[{d['dF_ci95'][0]:+.4f},{d['dF_ci95'][1]:+.4f}] "
                     f"每种子 {['%+.4f' % x for x in d['dF_vs_control_per_seed']]}")
    L.append("\n## persist 家族消融（none/tau2，k=1/2/3；trade-off 曲线）\n")
    L.append("| k | spectral switch mean/max | spectral F̄ | triplet F̄ | isolate 总数(spec) |")
    L.append("|---|---|---|---|---|")
    for k in [1, 2, 3]:
        lab = f"none/p{k}/tau2"
        if lab in tab:
            s, t = tab[lab]["spectral"], tab[lab]["triplet"]
            L.append(f"| {k} | {s['switch_mean']:.2f}/{s['switch_max']:.2f} | {s['F_mean']:.4f} | "
                     f"{t['F_mean']:.4f} | {s['n_isolate_total']} |")
    L.append("\n## F1 退化检查（跨种子汇总 Spearman，预注册口径；单种子仅参考）\n")
    L.append("| 规则 | 臂 | drift~bound | drift~beta | bound~beta | 汇总 max\\|ρ\\| | 触发 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in rules:
        for arm in ARMS:
            cs = []
            for p in [("drift", "bound"), ("drift", "beta"), ("bound", "beta")]:
                e = f1[lab][arm][f"{p[0]}~{p[1]}"]
                cs.append(f"{e['pooled_rho']}/去趋势{e['detrended_rho']}"
                          if "pooled_rho" in e else "constant")
            mx = max((abs(e["pooled_rho"]) for e in f1[lab][arm].values()
                      if "pooled_rho" in e), default=0.0)
            L.append(f"| {lab} | {arm} | " + " | ".join(cs) +
                     f" | {mx:.4f} | {'**是**' if mx > 0.9 else '否'} |")
    L.append("\n## 通道触发结构（决策点越界次数 z>tau；n_decision_points=5 种子×8）\n")
    L.append("| 规则 | 臂 | drift | bound | beta | 决策点数 |")
    L.append("|---|---|---|---|---|---|")
    for lab in rules:
        for arm in ARMS:
            f_ = fire[lab][arm]
            L.append(f"| {lab} | {arm} | {f_['drift']} | {f_['bound']} | {f_['beta']} | "
                     f"{f_['n_decision_points']} |")
    Path(args.md).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({
        "control": rule_label(CONTROL_KEY),
        "verdicts_primary": {r: v["pass_primary_max_reading"] for r, v in verdicts.items()},
        "verdicts_secondary": {r: v["pass_secondary_mean_reading"] for r, v in verdicts.items()},
        "f1_max_abs_pooled": f1_max,
        "spectral": {r: [tab[r]["spectral"]["F_mean"], tab[r]["spectral"]["switch_mean"],
                         tab[r]["spectral"]["switch_max"]] for r in rules},
        "triplet": {r: [tab[r]["triplet"]["F_mean"], tab[r]["triplet"]["switch_mean"],
                        tab[r]["triplet"]["switch_max"]] for r in rules},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
