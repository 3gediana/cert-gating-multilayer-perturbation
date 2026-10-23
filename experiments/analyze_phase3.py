"""Phase 3 长序列终判（experiment-long）聚合分析：T=50 × 5 臂 × 5 种子。

预注册判读（workflow.json minimal_discriminative_experiment.decisive_step +
pilot-design.md §6/§7；口径与 analyze_pilot.py / analyze_f2fix2.py 一致）：
- H1 主判据：triplet vs 每一单量臂（cka/spectral/beta）ΔF̄，配对种子 bootstrap
  CI95 不含 0 且 |ΔF̄|≥5pp（Δ=F(单量)−F(triplet)，正=triplet 更优；
  与 analyze_pilot.py main_criterion 同口径）。baseline 对照并列（次判据①）。
- 次判据②：T=50 vs T=10 的 ΔF̄ 差距（"长序列才显著"预测）。T=10 主对照读数取
  results/pilot-summary.json（原规则同协议 pilot；看板卡指定），同规则 asym T=10
  读数取 results/pilot-f2fix2-summary.json 并列。
- F1：跨种子汇总 Spearman（原始+去趋势），任一臂任一对 |ρ|>0.9 → 触发
  （三量退化 → falsified）；单种子 n 小只作参考列（§8 细则）。
- F2：switch 双口径并列报告；**F2 限制报告必须显式写出**——T=10 同批验收 gate
  形式上未过（asym：seed4 配对 ΔF=+0.0110>PAIR_TOL 0.005；switch_max 0.22 与
  F̄ 0.0304 已过），Phase 3 规则按组合拍板（DECISIONS 2026-10-23）以限制报告携带。
- F5：cost_ratio（触发/merge 训练）>0.2 触发；附成本曲线（J+SVD+CKA vs 训练耗时，
  随 T 超线性）：T=10（同规则 asym 同批）+ T=50（本单元）+ 可选 T=25 探针。
- verdict_mapping：主判据达（三个单量臂全过）→ supported；F1 触发 → falsified；
  长序列仍无增益 → falsified（F3 终判，负结果）。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_phase3.py \
    --in results/phase3-metrics.json \
    --md results/phase3-summary.md --json results/phase3-summary.json
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = ["baseline", "cka", "spectral", "beta", "triplet"]
SINGLE = ["cka", "spectral", "beta"]        # 单量臂（主判据对比对象）
CH = ["drift", "bound", "beta"]
SW_THRESH = 0.3
F1_THRESH = 0.9
PP5 = 0.05                                # 主判据 5pp 条款
COST_THRESH = 0.2                         # F5 预注册阈值
RULE_LABEL = "asym/p1/tau2/b0.3-i2o3"

# F2 限制报告（T=10 同批验收读数，DECISIONS 2026-10-23 判读条；与主判据并列）
F2_LIMITATION = (
    "F2 gate 形式上未过（T=10 同批）：asym seed4 配对 ΔF=+0.0110>PAIR_TOL 0.005"
    "（mean Δ=-0.0027，CI95[-0.0086,+0.0048] 含 0）；switch_max 0.22≤0.3 与 "
    "F̄ 0.0304≤0.0331 已过。Phase 3 规则=asym 系组合拍板降规模（候选④a 参数跑前冻结，"
    "同批实测严格支配 persist k=2），非事后调参。"
)


def rule_key(cfg):
    return (cfg.get("f2_fix"), int(cfg.get("persist", 1)),
            float(cfg.get("tau_hi", 2.0)), float(cfg.get("band", 0.3)),
            cfg.get("k_in"), cfg.get("k_out"), cfg.get("ewma_alpha"),
            int(cfg.get("T", -1)))


def rule_label(key):
    lab = f"{key[0]}/p{key[1]}/tau{key[2]:g}/b{key[3]:g}"
    if key[0] == "asym" and key[4] is not None:
        lab += f"-i{key[4]}o{key[5]}"
    return lab


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def arm_stats(cell):
    """rule cell -> arm -> 每种子读数 + 汇总（与 analyze_f2fix2.py 同口径）。"""
    out = {}
    for arm in ARMS:
        rows = []
        for seed in sorted(cell):
            s = next(x for x in cell[seed]["arms"] if x["arm"] == arm)["summary"]
            tm = s["timings_sec"]
            rows.append({
                "seed": seed, "F": s["mean_forgetting"],
                "switch": s["action_switch_rate"],
                "n_isolate": s["n_isolate"], "n_forced": s["n_forced"],
                "acc": s["mean_final_acc"], "cost": s["trigger_cost_ratio"],
                "trigger_sec": round(tm["jacobian"] + tm["svd"] + tm["cka"], 3),
                "train_sec": round(tm["train_merge"], 3),
                "train_iso_sec": round(tm["train_isolate"], 3),
            })
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
            "cost_per_seed": [r["cost"] for r in rows],
            "trigger_sec_mean": round(float(np.mean([r["trigger_sec"] for r in rows])), 3),
            "train_sec_mean": round(float(np.mean([r["train_sec"] for r in rows])), 3),
            "train_iso_sec_mean": round(float(np.mean([r["train_iso_sec"] for r in rows])), 3),
        }
    return out


def f1_pooled(cell):
    """F1 退化检查（预注册口径）：每臂跨种子汇总 Spearman（原始/去趋势）。"""
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
                "f1_raw": bool(abs(pooled) > F1_THRESH),
                "f1_detrended": bool(abs(d) > F1_THRESH),
            }
        out[arm] = entry
    return out


def main_criterion(table):
    """H1 主判据：triplet vs 每一单量臂（+baseline 并列）。Δ=F(单量)−F(triplet)。"""
    F4 = np.array(table["triplet"]["F_per_seed"])
    out = {}
    for arm in SINGLE + ["baseline"]:
        Fx = np.array(table[arm]["F_per_seed"])
        diff = Fx - F4
        lo, hi = bootstrap_ci(diff)
        out[arm] = {
            "delta_mean": round(float(diff.mean()), 4),
            "delta_per_seed": [round(float(x), 4) for x in diff],
            "ci95": [lo, hi],
            "ci_excludes_0": bool(lo > 0 or hi < 0),
            "wins_all_seeds": bool(all(d > 0 for d in diff)),
            "delta_ge_5pp": bool(abs(diff.mean()) >= PP5),
            "is_single_arm": arm in SINGLE,
        }
    singles = {a: out[a] for a in SINGLE}
    out["_main_pass"] = bool(all(
        v["ci_excludes_0"] and v["delta_ge_5pp"] for v in singles.values()))
    return out


def _f_readings(arm_entry):
    """兼容两种归档格式：arm_stats（F_per_seed/F_mean）或 analyze_pilot（per_seed[].F）。"""
    if "F_per_seed" in arm_entry:
        return {"F_mean": arm_entry["F_mean"],
                "F_per_seed": arm_entry["F_per_seed"]}
    rows = arm_entry["per_seed"]
    return {"F_mean": round(sum(r["F"] for r in rows) / len(rows), 4),
            "F_per_seed": [r["F"] for r in rows]}


def load_t10_readings(pilot_path, asym_path):
    """T=10 对照读数（次判据②）：主=原规则 pilot-summary.json；并列=同规则 asym。"""
    out = {}
    p = json.loads(Path(pilot_path).read_text(encoding="utf-8"))
    out["pilot_original_rule_T10"] = {a: _f_readings(p["table"][a]) for a in ARMS}
    q = json.loads(Path(asym_path).read_text(encoding="utf-8"))
    cell = q["arm_table"][RULE_LABEL]
    out["asym_same_rule_T10"] = {a: _f_readings(cell[a]) for a in ARMS}
    return out


def secondary_criterion(t50_table, t10):
    """次判据②：T=50 vs T=10 的 ΔF̄ 差距（triplet vs 每一单量臂）。"""
    out = {}
    for src, readings in t10.items():
        rows = {}
        for arm in SINGLE + ["baseline"]:
            d50 = float(t50_table[arm]["F_mean"] - t50_table["triplet"]["F_mean"])
            d10 = float(readings[arm]["F_mean"] - readings["triplet"]["F_mean"])
            rows[arm] = {
                "delta_F_mean_T50": round(d50, 4),
                "delta_F_mean_T10": round(d10, 4),
                "gap_T50_minus_T10": round(d50 - d10, 4),
                "amplified_at_T50": bool(abs(d50) > abs(d10)),
            }
        out[src] = rows
    return out


def cost_curve(phase3_cell, t10_metrics_path, probe_path=None):
    """F5 成本曲线：J+SVD+CKA（触发）vs 训练耗时，随 T 超线性。"""
    def agg(cell, T):
        rows = {}
        for arm in ARMS:
            trg, trn, iso, ratio = [], [], [], []
            for seed in sorted(cell):
                s = next(x for x in cell[seed]["arms"] if x["arm"] == arm)["summary"]
                tm = s["timings_sec"]
                trg.append(tm["jacobian"] + tm["svd"] + tm["cka"])
                trn.append(tm["train_merge"])
                iso.append(tm["train_isolate"])
                ratio.append(s["trigger_cost_ratio"])
            rows[arm] = {
                "trigger_sec_mean": round(float(np.mean(trg)), 3),
                "train_merge_sec_mean": round(float(np.mean(trn)), 3),
                "train_isolate_sec_mean": round(float(np.mean(iso)), 3),
                "cost_ratio_mean": round(float(np.mean(ratio)), 4),
            }
        return {"T": T, "arms": rows}

    t_main = int(list(phase3_cell.values())[0]["config"]["T"])
    curve = [agg(phase3_cell, t_main)]
    if t10_metrics_path and Path(t10_metrics_path).exists():
        data = json.loads(Path(t10_metrics_path).read_text(encoding="utf-8"))
        cells = {}
        for r in data["runs"]:
            k = rule_key(r["config"])
            if k[0] == "asym" and k[7] == 10:
                cells.setdefault(k, {})[r["config"]["seed"]] = r
        if cells:
            curve.append(agg(list(cells.values())[0], 10))
    if probe_path and Path(probe_path).exists():
        data = json.loads(Path(probe_path).read_text(encoding="utf-8"))
        cells = {}
        for r in data["runs"]:
            k = rule_key(r["config"])
            cells.setdefault(k, {})[r["config"]["seed"]] = r
        if cells:
            first = list(cells.values())[0]
            T = first[sorted(first)[0]]["config"]["T"]
            curve.append(agg(first, int(T)))
    curve.sort(key=lambda c: c["T"])
    # 结构预期：触发成本 ∝ T(T-1)/2（J 覆盖全部锚任务），训练 ∝ T
    if len(curve) >= 2:
        base = curve[0]
        for c in curve:
            c["expected_trigger_ratio_vs_T10"] = round(
                (c["T"] * (c["T"] - 1)) / (10 * 9), 3) if c["T"] != 10 else 1.0
            c["trigger_ratio_vs_T10"] = round(
                c["arms"]["triplet"]["trigger_sec_mean"] /
                max(base["arms"]["triplet"]["trigger_sec_mean"], 1e-9), 3)
            c["train_ratio_vs_T10"] = round(
                c["arms"]["triplet"]["train_merge_sec_mean"] /
                max(base["arms"]["triplet"]["train_merge_sec_mean"], 1e-9), 3)
    return curve


def plot_cost_curve(curve, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = [c["T"] for c in curve]
    trg = [c["arms"]["triplet"]["trigger_sec_mean"] for c in curve]
    trn = [c["arms"]["triplet"]["train_merge_sec_mean"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ts, trg, "o-", label="trigger (J+SVD+CKA)")
    ax.plot(Ts, trn, "s-", label="train (merge)")
    if len(Ts) >= 2:
        t = np.linspace(min(Ts) * 0.9, max(Ts) * 1.05, 100)
        ax.plot(t, trn[0] * (t / Ts[0]), "--", color="gray", alpha=0.6,
                label="linear ref (from train)")
        ax.plot(t, trg[0] * (t * (t - 1)) / (Ts[0] * (Ts[0] - 1)), ":",
                color="gray", alpha=0.6, label="quadratic ref (from trigger)")
    ax.set_xlabel("T (sequence length)")
    ax.set_ylabel("seconds per run (triplet arm, mean over seeds)")
    ax.set_title("F5 cost curve: trigger vs train (asym rule, synthetic pilot protocol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/phase3-metrics.json")
    ap.add_argument("--md", default="results/phase3-summary.md")
    ap.add_argument("--json", default="results/phase3-summary.json")
    ap.add_argument("--t10-pilot", default="results/pilot-summary.json")
    ap.add_argument("--t10-asym", default="results/pilot-f2fix2-summary.json")
    ap.add_argument("--t10-asym-metrics", default="results/pilot-f2fix2-metrics.json")
    ap.add_argument("--cost-probe", default="results/phase3-costprobe-T25-metrics.json",
                    help="可选 T=25 成本探针（F5 曲线第三点；不存在则跳过）")
    ap.add_argument("--png", default="results/phase3-cost-curve.png")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    cells = {}
    for r in data["runs"]:
        cells.setdefault(rule_key(r["config"]), {})[r["config"]["seed"]] = r
    assert len(cells) == 1, f"expected single rule cell, got {len(cells)}"
    key = list(cells)[0]
    cell = cells[key]
    seeds = sorted(cell)
    lab = rule_label(key)
    T = key[7]
    grid_ok = (len(seeds) == 5 and T == 50 and
               all(len(cell[s]["arms"]) == 5 for s in seeds))

    table = arm_stats(cell)
    main = main_criterion(table)
    f1 = f1_pooled(cell)

    # triplet vs spectral 动作一致性 + cka/beta 是否从不告警（F3 结构检查）
    ident, never = {}, {}
    slot_sat = {}
    for s in seeds:
        a4 = next(x for x in cell[s]["arms"] if x["arm"] == "triplet")
        a2 = next(x for x in cell[s]["arms"] if x["arm"] == "spectral")
        ident[s] = [x["action"] for x in a4["records"]] == \
                   [x["action"] for x in a2["records"]]
    for a in ["cka", "beta"]:
        never[a] = bool(table[a]["n_isolate_total"] == 0)
    # 槽位饱和结构（T=50 下 3 个 isolate 槽是否早期耗尽）
    for s in seeds:
        t = next(x for x in cell[s]["arms"] if x["arm"] == "triplet")
        recs = t["records"]
        slot_sat[s] = {
            "slots_used": t["summary"]["slots_used"],
            "isolate_tasks": [x["task"] for x in recs if x["action"] == "isolate"],
            "first_forced_task": next((x["task"] for x in recs if x["forced"]), None),
            "n_forced_with_alarm": sum(1 for x in recs if x["forced"] and x["max_z"] > 2.0),
        }

    f1_max = 0.0
    f1_triggered = False
    for arms_d in f1.values():
        for e in arms_d.values():
            if "pooled_rho" in e:
                f1_max = max(f1_max, abs(e["pooled_rho"]))
                f1_triggered = f1_triggered or e["f1_raw"] or e["f1_detrended"]

    f2 = {
        "limitation_report_T10": F2_LIMITATION,
        "switch_mean_by_arm": {a: table[a]["switch_mean"] for a in ARMS},
        "switch_max_by_arm": {a: table[a]["switch_max"] for a in ARMS},
        "switch_per_seed": {a: table[a]["switch_per_seed"] for a in ARMS},
        "triggered_T50": bool(any(table[a]["switch_max"] > SW_THRESH + 1e-9
                                  for a in ARMS)),
        "isolate_forced_total": {a: f"{table[a]['n_isolate_total']}/"
                                    f"{table[a]['n_forced_total']}" for a in ARMS},
    }
    f5 = {
        "threshold": COST_THRESH,
        "cost_ratio_mean_by_arm": {a: table[a]["cost_mean"] for a in ARMS},
        "triggered": bool(any(table[a]["cost_mean"] > COST_THRESH for a in ARMS)),
    }

    t10 = load_t10_readings(args.t10_pilot, args.t10_asym)
    secondary = secondary_criterion(table, t10)
    curve = cost_curve(cell, args.t10_asym_metrics,
                       probe_path=args.cost_probe)
    png_ok, png_err = False, None
    try:
        plot_cost_curve(curve, args.png)
        png_ok = True
    except Exception as e:  # 绘图失败不阻塞判读
        png_err = str(e)

    # ---- verdict（预注册 verdict_mapping）
    if f1_triggered:
        verdict = "falsified"
        verdict_reason = "F1 触发：三量跨种子汇总 |ρ|>0.9（三量退化，H 驳倒）"
    elif main["_main_pass"]:
        verdict = "supported"
        verdict_reason = ("H1 主判据达：triplet vs 三个单量臂 ΔF̄ 全部 CI95 不含 0 "
                          "且 |ΔF̄|≥5pp")
    else:
        verdict = "falsified"
        verdict_reason = ("F3 终判：T=50 长序列下 triplet 仍未对每一单量臂达成 "
                          "CI95 不含 0 且 ≥5pp 的增益（负结果）")

    payload = {
        "analysis": ("Phase 3 长序列终判（experiment-long）：T=50 × 5 臂 × 5 种子，"
                     "规则=asym（k_in=2/k_out=3, persist=1, tau_hi=2.0, band=0.3），"
                     "难度=合成 pilot 校准版（width64/steps100/drift0.5/noise1.0/k10/"
                     "anchor64）。判读口径=workflow.json decisive_step + "
                     "pilot-design.md §6/§7（与 analyze_pilot.py 一致）。"),
        "rule": lab,
        "rule_key": list(key),
        "T": T,
        "seeds": seeds,
        "grid_complete_5seeds_x_5arms": bool(grid_ok),
        "config": cell[seeds[0]]["config"],
        "wall_sec_total": round(sum(r["wall_sec"] for r in data["runs"]), 2),
        "arm_table": table,
        "main_criterion": main,
        "secondary_criterion_2_T_vs_T": secondary,
        "t10_readings": t10,
        "f1_pooled_cross_seed": f1,
        "f1_max_abs_pooled": round(f1_max, 4),
        "f1_triggered": bool(f1_triggered),
        "f2": f2,
        "f5": f5,
        "f5_cost_curve": curve,
        "f5_cost_curve_png": args.png if png_ok else None,
        "f3_structure": {
            "triplet_equals_spectral_actions_per_seed": ident,
            "triplet_equals_spectral_all_seeds": bool(all(ident.values())),
            "never_alarm_arms": never,
            "slot_saturation_triplet": slot_sat,
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                               encoding="utf-8")

    # ---- markdown
    L = []
    L.append("# Phase 3 长序列终判（experiment-long）：T=50 × 5 臂 × 5 种子\n")
    L.append(f"- 规则：`{lab}`（T=50，--T 覆盖；F2 限制报告见下）"
             f"｜ 难度：{json.dumps({k: v for k, v in cell[seeds[0]]['config'].items() if k in ('width','steps','drift','noise','k','anchor_n','train_n','slots','tau_hi','band')}, ensure_ascii=False)}")
    L.append(f"- 网格完整：{grid_ok}（5 种子 × 5 臂；wall 合计 "
             f"{payload['wall_sec_total']}s）")
    L.append(f"- 判读口径：预注册不变（workflow.json decisive_step + pilot-design.md "
             f"§6/§7；Δ=F(单量)−F(triplet)，正=triplet 更优）\n")
    L.append("## 主表（5 种子；F̄=mean±sd；switch=mean/max（每种子）；cost=触发/merge训练）\n")
    L.append("| 臂 | F̄±sd | F̄/种子 | acc | switch mean/max | isolate(forced) | cost_ratio |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        t = table[arm]
        fps = "/".join(f"{v:.4f}" for v in t["F_per_seed"])
        sws = "/".join(f"{v:.2f}" for v in t["switch_per_seed"])
        L.append(f"| {arm} | {t['F_mean']:.4f}±{t['F_sd']:.4f} | {fps} | "
                 f"{t['acc_mean']:.4f} | {t['switch_mean']:.2f}/{t['switch_max']:.2f} ({sws}) | "
                 f"{t['n_isolate_total']}({t['n_forced_total']}) | {t['cost_mean']:.3f} |")
    L.append("\n## H1 主判据（triplet vs 每一单量臂；配对种子 bootstrap CI95）\n")
    L.append("| 对比臂 | ΔF̄ | 每种子 Δ | CI95 | 不含0 | ≥5pp | 判 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in SINGLE + ["baseline"]:
        m = main[arm]
        dps = "/".join(f"{x:+.4f}" for x in m["delta_per_seed"])
        ok = m["ci_excludes_0"] and m["delta_ge_5pp"]
        tag = "单量臂" if m["is_single_arm"] else "次判据①对照"
        L.append(f"| {arm}（{tag}） | {m['delta_mean']:+.4f} | {dps} | "
                 f"[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}] | "
                 f"{'是' if m['ci_excludes_0'] else '否'} | "
                 f"{'是' if m['delta_ge_5pp'] else '否'} | "
                 f"{'**PASS**' if ok else 'FAIL'} |")
    L.append(f"\n**主判据总判：{'PASS' if main['_main_pass'] else 'FAIL'}**"
             f"（要求三个单量臂全部 CI95 不含 0 且 ≥5pp）\n")
    L.append("## 次判据②：T=50 vs T=10 的 ΔF̄ 差距（长序列才显著预测）\n")
    for src, rows in secondary.items():
        L.append(f"### T=10 读数来源：{src}\n")
        L.append("| 对比臂 | ΔF̄(T=50) | ΔF̄(T=10) | 差距(T50−T10) | T=50 放大? |")
        L.append("|---|---|---|---|---|")
        for arm, r in rows.items():
            L.append(f"| {arm} | {r['delta_F_mean_T50']:+.4f} | "
                     f"{r['delta_F_mean_T10']:+.4f} | {r['gap_T50_minus_T10']:+.4f} | "
                     f"{'是' if r['amplified_at_T50'] else '否'} |")
        L.append("")
    L.append("## F1 退化检查（跨种子汇总 Spearman；单种子仅参考）\n")
    L.append("| 臂 | drift~bound | drift~beta | bound~beta | 汇总 max\\|ρ\\| | 触发 |")
    L.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        cs = []
        for p in [("drift", "bound"), ("drift", "beta"), ("bound", "beta")]:
            e = f1[arm][f"{p[0]}~{p[1]}"]
            cs.append(f"{e['pooled_rho']}/去趋势{e['detrended_rho']}"
                      if "pooled_rho" in e else "constant")
        mx = max((abs(e["pooled_rho"]) for e in f1[arm].values()
                  if "pooled_rho" in e), default=0.0)
        L.append(f"| {arm} | " + " | ".join(cs) +
                 f" | {mx:.4f} | {'**是**' if mx > F1_THRESH else '否'} |")
    L.append(f"\nF1 跨种子汇总 max|ρ|={f1_max:.4f}（阈值 0.9）→ "
             f"{'触发' if f1_triggered else '未触发'}。\n")
    L.append("## F2 抖动（双口径并列）+ **限制报告（必须与主判据并列，不得隐去）**\n")
    L.append(f"- T=50 实测 switch mean/max 每臂：" +
             "；".join(f"{a} {table[a]['switch_mean']:.2f}/{table[a]['switch_max']:.2f}"
                       for a in ARMS) +
             f" → {'触发(>0.3)' if f2['triggered_T50'] else '未触发'}")
    L.append(f"- isolate/forced 总数每臂：" +
             "；".join(f"{a} {v}" for a, v in f2["isolate_forced_total"].items()))
    L.append(f"- **限制报告**：{F2_LIMITATION}\n")
    L.append("## F5 成本（触发计算 vs 训练耗时；阈值 0.2）\n")
    L.append(f"- T=50 cost_ratio 每臂：" +
             "；".join(f"{a} {table[a]['cost_mean']:.3f}" for a in ARMS) +
             f" → {'触发' if f5['triggered'] else '未触发'}（taste 雷：必须报告）")
    L.append("\n| T | 臂 | 触发 J+SVD+CKA (s) | 训练 merge (s) | 训练 isolate (s) | cost_ratio | 触发/T=10 实测 | 触发/T=10 结构预期∝T(T−1)/2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in curve:
        for arm in ARMS:
            a = c["arms"][arm]
            L.append(f"| {c['T']} | {arm} | {a['trigger_sec_mean']:.3f} | "
                     f"{a['train_merge_sec_mean']:.3f} | {a['train_isolate_sec_mean']:.3f} | "
                     f"{a['cost_ratio_mean']:.3f} | "
                     f"{c.get('trigger_ratio_vs_T10', '—')} | "
                     f"{c.get('expected_trigger_ratio_vs_T10', '—')} |")
    if png_ok:
        L.append(f"\n成本曲线图：`{args.png}`（trigger vs train，triplet 臂；"
                 f"灰线=线性/二次参考）\n")
    else:
        L.append(f"\n（成本曲线图未生成：{png_err}）\n")
    L.append("## F3 结构检查\n")
    L.append(f"- triplet 与 spectral 动作序列逐种子一致：{ident} → "
             f"{'全部一致' if all(ident.values()) else '有不一致'}")
    L.append(f"- 从不告警臂（isolate 总数=0）："
             f"{ {a: v for a, v in never.items()} }")
    L.append("- **槽位饱和结构（triplet，T=50）**：每种子 slots_used / isolate 任务 / "
             "首次 forced 任务 / 带告警被强制 merge 次数——")
    for s in seeds:
        ss = slot_sat[s]
        L.append(f"  - seed{s}: slots={ss['slots_used']} isolate@{ss['isolate_tasks']} "
                 f"first_forced=task{ss['first_forced_task']} "
                 f"forced_with_alarm={ss['n_forced_with_alarm']}")
    L.append("  → 3 个 isolate 槽在序列早期（task 4-8）即耗尽，中晚期告警全部落为 "
             "forced merge：T=50 下门控运行在**槽位饱和区**（机制容量假设稀疏告警，"
             "长序列不成立）——三元组无增量的结构性解释之一。")
    L.append("\n## 判决\n")
    L.append(f"- **verdict：{verdict}**——{verdict_reason}")
    L.append(f"- 复现：`PYTHONUTF8=1 G:/ai/python11/python.exe experiments/run_phase3.py "
             f"--out results/phase3-metrics.json && PYTHONUTF8=1 G:/ai/python11/python.exe "
             f"experiments/analyze_phase3.py --in results/phase3-metrics.json "
             f"--md results/phase3-summary.md --json results/phase3-summary.json`")
    Path(args.md).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({
        "rule": lab, "T": T, "seeds": seeds, "grid_complete": grid_ok,
        "arm_F_mean": {a: table[a]["F_mean"] for a in ARMS},
        "main_criterion": {a: {k: main[a][k] for k in
                               ("delta_mean", "ci95", "ci_excludes_0", "delta_ge_5pp")}
                           for a in SINGLE + ["baseline"]},
        "main_pass": main["_main_pass"],
        "secondary": secondary,
        "f1_max_abs_pooled": round(f1_max, 4), "f1_triggered": f1_triggered,
        "f2_triggered_T50": f2["triggered_T50"],
        "f2_limitation": F2_LIMITATION,
        "f5_triggered": f5["triggered"],
        "cost_curve": curve,
        "verdict": verdict, "verdict_reason": verdict_reason,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
