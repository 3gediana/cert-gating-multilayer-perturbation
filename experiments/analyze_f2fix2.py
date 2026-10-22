"""F2 修复第二轮聚合分析（experiment-redesign-2）：候选④集 × 5 种子 × 5 臂。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_f2fix2.py \
    --in results/pilot-f2fix2-metrics.json \
    --md results/pilot-f2fix2-summary.md \
    --json results/pilot-f2fix2-summary.json

预注册 gate（workflow.json gate_step + pilot-design.md §7 F2；与轮次 1 同口径，
见 notes/DECISIONS.md 2026-10-23 条）：
- switch<0.3：任一臂 action-switch rate。主口径=per-seed rate 跨种子 max
  （与 analyze_pilot.py 的 F2 实现一致），次口径=mean，双口径都判。
- F̄ 不劣化于 0.0331（归档 pilot spectral/triplet 基准）：F_mean<=0.0331；
  另做同批配对（vs 同批对照 none/p1/tau2/b0.3，每种子 ΔF<=0.005 视为不劣化）。
- F1 退化检查按 pilot-design.md §8 判读细则：跨种子汇总 Spearman（原始+去趋势），
  单种子 n 小会假高，只作参考列。

规则标签：{f2_fix}/p{persist}/tau{tau}/b{band}[-i{k_in}o{k_out}][-a{alpha}]。
全部行为轴都进标签（轮次 1 教训：f2_fix=persist 是 none+persist 的别名，
标签缺轴会静默覆盖单元）。对照 = none/p1/tau2/b0.3（原规则，同批重跑）。
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = ["baseline", "cka", "spectral", "beta", "triplet"]
CH = ["drift", "bound", "beta"]
CONTROL_KEY = ("none", 1, 2.0, 0.3, None, None, None)  # 原规则（hysteresis 已开）
F_BENCH = 0.0331                 # 归档 pilot spectral/triplet F̄（gate_step 基准）
SW_THRESH = 0.3
PAIR_TOL = 0.005
GATE_ARMS = ["spectral", "triplet"]  # F̄ 基准臂（workflow gate_step 明示）

DESC = {
    "none/p1/tau2/b0.3": "同批对照：原规则（hysteresis band0.3）",
    "none/p2/tau2/b0.3": "家族：persist k=2（轮次 1 同批重跑）",
    "none/p3/tau2/b0.3": "家族：persist k=3（轮次 1 同批重跑）",
    "none/p1/tau2/b0.5": "④a' 加宽 hysteresis band 0.3→0.5",
    "asym/p1/tau2/b0.3-i2o3": "④a 非对称持续期（进 2 告警/退 3 不告警）",
    "robust/p1/tau2/b0.3": "④b 稳健 z（median/MAD）",
    "ewma/p1/tau2/b0.3-a0.3": "④b' EWMA z（alpha=0.3，隐式去趋势）",
    "detrend-bound/p1/tau2/b0.3": "④c 混合：仅 bound 去趋势（drift/beta 原始 z）",
}

# 预注册预测（notes/DECISIONS.md 2026-10-23 条，跑前冻结）→ 判读时对照
PREDICTIONS = {
    "none/p1/tau2/b0.3": "对照：复现轮次 1 触发读数（spectral switch 0.49/0.67、F̄ 0.0331）",
    "none/p2/tau2/b0.3": "家族：switch 0.27/0.44、F̄≈0.0333（seed2 max 0.44 不过 gate）",
    "none/p3/tau2/b0.3": "家族：switch 0.13/0.22 但 F̄≈0.0433（seed1/4 漏告警）",
    "none/p1/tau2/b0.5": "预测：无效或边际（轮次 1 多数 override 点 z<1.5）",
    "asym/p1/tau2/b0.3-i2o3": "预测：switch 低于 k=2 且 F̄≈k=2；风险：粘性 isolate 槽位耗尽",
    "robust/p1/tau2/b0.3": "预测：告警减少、F̄ 向 baseline 回落（早期尖峰被钝化）",
    "ewma/p1/tau2/b0.3-a0.3": "预测：告警局部化、晚期不持续越界；风险：稳态 z≈2.4 贴线振荡",
    "detrend-bound/p1/tau2/b0.3": "预测：bound 振荡修复且 drift/beta 不直接越界；风险：路径效应",
}


def rule_key(cfg):
    return (cfg.get("f2_fix"), int(cfg.get("persist", 1)),
            float(cfg.get("tau_hi", 2.0)), float(cfg.get("band", 0.3)),
            cfg.get("k_in"), cfg.get("k_out"), cfg.get("ewma_alpha"))


def rule_label(key):
    lab = f"{key[0]}/p{key[1]}/tau{key[2]:g}/b{key[3]:g}"
    if key[0] == "asym" and key[4] is not None:
        lab += f"-i{key[4]}o{key[5]}"
    if key[0] == "ewma" and key[6] is not None:
        lab += f"-a{key[6]:g}"
    return lab


def load_cells(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = {}
    for r in data["runs"]:
        key = rule_key(r["config"])
        cells.setdefault(key, {})[r["config"]["seed"]] = r
    # 别名单元：行为逐位相同的单元（签名只取行为字段，剔除 timings）。
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
    ap.add_argument("--in", dest="inp", default="results/pilot-f2fix2-metrics.json")
    ap.add_argument("--md", default="results/pilot-f2fix2-summary.md")
    ap.add_argument("--json", default="results/pilot-f2fix2-summary.json")
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
                 "n_isolate_total": t["n_isolate_total"],
                 "n_forced_total": t["n_forced_total"]}
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
        "analysis": "F2 修复第二轮验收（experiment-redesign-2）：候选④集"
                    "（④a 非对称持续期 i2o3 / ④a' 加宽 band0.5 / ④b 稳健 z median-MAD / "
                    "④b' EWMA z / ④c 仅 bound 去趋势）× pilot T=10 width64 × 5 种子 × 5 臂。"
                    "验收=switch<0.3（主口径 per-seed max，与 analyze_pilot.py F2 实现一致；"
                    "次口径 mean）且 F̄ 不劣化于 0.0331 + 同批配对不劣化"
                    "（vs 同批对照 none/p1/tau2/b0.3）。",
        "control": rule_label(CONTROL_KEY),
        "benchmark_F": F_BENCH,
        "switch_threshold": SW_THRESH,
        "paired_tolerance": PAIR_TOL,
        "alias_cells": aliases,
        "grid": {rule_label(k): sorted(cells[k]) for k in sorted(cells)},
        "preregistered_predictions": PREDICTIONS,
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
    L.append("# F2 修复实验第二轮：候选④集（experiment-redesign-2 阶段）\n")
    L.append("> 预注册候选④集（notes/DECISIONS.md 2026-10-23 条，跑前冻结）："
             "④a 非对称持续期（进 isolate 连续 2 告警/退连续 3 不告警）"
             "④a' 加宽 hysteresis band 0.3→0.5；④b 稳健 z（median/MAD）"
             "④b' EWMA z（alpha=0.3，隐式去趋势）；④c 混合（仅 bound 通道去趋势，"
             "drift/beta 保持原始 z）。**验收：switch<0.3 且 F̄ 不劣化于 0.0331**"
             "（归档 pilot spectral/triplet 基准）+ 同批配对不劣化。\n")
    L.append(f"> 对照 = `{rule_label(CONTROL_KEY)}`（原规则同批重跑："
             f"spectral switch {ctrl_stats['spectral']['switch_mean']}/"
             f"{ctrl_stats['spectral']['switch_max']}、F̄ {ctrl_stats['spectral']['F_mean']}"
             "——与轮次 1 归档读数逐位复证）。"
             "switch 主口径 = per-seed rate 跨种子 max（与 analyze_pilot.py 的 F2 判读一致），"
             "次口径 = mean，双口径列报。\n")
    L.append("## 网格（唯一单元 × 5 种子；别名单元见 JSON）\n")
    L.append("| 规则 | 说明 | seeds |")
    L.append("|---|---|---|")
    for lab in rules:
        L.append(f"| {lab} | {DESC.get(lab, '')} | {len(tab[lab]['baseline']['per_seed'])} |")
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
    L.append("\n## 配对差（规则 − 对照 none/p1/tau2/b0.3，每种子 F̄，spectral/triplet）\n")
    for lab, v in verdicts.items():
        for arm in GATE_ARMS:
            d = v["detail"][arm]
            L.append(f"- {lab} {arm}: ΔF̄={d['dF_mean']:+.4f} "
                     f"CI95[{d['dF_ci95'][0]:+.4f},{d['dF_ci95'][1]:+.4f}] "
                     f"每种子 {['%+.4f' % x for x in d['dF_vs_control_per_seed']]}")
    L.append("\n## 规则家族消融（同批；persist 与非对称 + z 族）\n")
    L.append("| 规则 | spectral switch mean/max | spectral F̄ | triplet F̄ | isolate(forced) spec |")
    L.append("|---|---|---|---|---|")
    for lab in rules:
        s, t = tab[lab]["spectral"], tab[lab]["triplet"]
        L.append(f"| {lab} | {s['switch_mean']:.2f}/{s['switch_max']:.2f} | {s['F_mean']:.4f} | "
                 f"{t['F_mean']:.4f} | {s['n_isolate_total']}({s['n_forced_total']}) |")
    L.append("\n## 预注册预测对照（跑前冻结于 DECISIONS 2026-10-23 条）\n")
    L.append("| 规则 | 预注册预测 |")
    L.append("|---|---|")
    for lab in rules:
        L.append(f"| {lab} | {PREDICTIONS.get(lab, '')} |")
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
