"""Phase 1 pilot 聚合分析：每臂遗忘/三量两两 Spearman（F1 退化检查）/switch rate/
成本比表 + 预注册判读（pilot-design.md §6-§7）。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py \
    --in results/pilot-metrics.json --md results/pilot-summary.md \
    --json results/pilot-summary.json

判读规则（预注册）:
- 主判据: F(A4) < F(A1/A2/A3) 各自；配对种子 bootstrap 95% CI 不含 0；ΔF̄ ≥ 5 分
- F1: 任一臂三量两两 |ρ|>0.9 -> 三元组退化为单量
- F2: 任一臂 action-switch rate > 30%
- F5: 触发计算 > 单任务训练耗时 20%（cost_ratio > 0.2）
注：F1 在原始序列上按预注册口径计算，同时报告去趋势稳健性（去线性趋势后
汇总 Spearman）——三量随任务序号有单调趋势，原始口径可能高估 |ρ|。
"""
import argparse
import json
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diff), len(diff))
        diffs.append(np.mean(diff[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def detrended_pooled(series_by_seed):
    """各种子序列去掉任务序号的线性趋势后汇总（趋势相关性稳健性检查）。"""
    out = []
    for vals in series_by_seed:
        x = np.arange(1, len(vals) + 1, dtype=float)
        v = np.asarray(vals, dtype=float)
        if len(v) > 2 and v.std() > 0:
            b, a = np.polyfit(x, v, 1)
            out.append(v - (a + b * x))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/pilot-metrics.json")
    ap.add_argument("--md", default="results/pilot-summary.md")
    ap.add_argument("--json", default="results/pilot-summary.json")
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))
    runs = data["runs"]
    seeds = [r["config"]["seed"] for r in runs]
    arms = [a["arm"] for a in runs[0]["arms"]]
    CH = ["drift", "bound", "beta"]

    # ---- 每臂 × 种子汇总表
    table = {}
    for arm in arms:
        rows = []
        for r in runs:
            a = next(x for x in r["arms"] if x["arm"] == arm)
            s = a["summary"]
            rows.append({
                "seed": r["config"]["seed"],
                "F": s["mean_forgetting"],
                "acc": s["mean_final_acc"],
                "n_merge": s["n_merge"], "n_isolate": s["n_isolate"],
                "n_forced": s["n_forced"],
                "switch": s["action_switch_rate"],
                "cost_ratio": s["trigger_cost_ratio"],
                "corr": {k: v["rho"] for k, v in s["correlations"].items()},
                "per_task_F": s["per_task_forgetting"],
            })
        Fs = np.array([x["F"] for x in rows])
        table[arm] = {
            "per_seed": rows,
            "F_mean": round(float(Fs.mean()), 4),
            "F_sd": round(float(Fs.std(ddof=1)), 4),
            "acc_mean": round(float(np.mean([x["acc"] for x in rows])), 4),
            "switch_mean": round(float(np.mean([x["switch"] for x in rows])), 4),
            "switch_max": round(float(np.max([x["switch"] for x in rows])), 4),
            "n_isolate_total": int(sum(x["n_isolate"] for x in rows)),
            "n_forced_total": int(sum(x["n_forced"] for x in rows)),
            "cost_mean": round(float(np.mean([x["cost_ratio"] for x in rows])), 4),
        }

    # ---- F1 退化检查：每臂跨种子汇总（原始 + 去趋势）
    corr_report = {}
    for arm in arms:
        pooled_raw = {p: [] for p in combinations(CH, 2)}
        per_seed = {p: [] for p in combinations(CH, 2)}
        for r in runs:
            a = next(x for x in r["arms"] if x["arm"] == arm)
            series = {c: [rec["raws"][c] for rec in a["records"][1:]]
                      for c in CH}
            for p in combinations(CH, 2):
                xa, xb = series[p[0]], series[p[1]]
                if len(set(xa)) <= 1 or len(set(xb)) <= 1:
                    per_seed[p].append({"seed": r["config"]["seed"], "rho": None,
                                        "constant": True})
                    continue
                rho = float(spearmanr(xa, xb).correlation)
                per_seed[p].append({"seed": r["config"]["seed"],
                                    "rho": round(rho, 4)})
                pooled_raw[p].append((xa, xb))
        entry = {}
        for p in combinations(CH, 2):
            key = f"{p[0]}~{p[1]}"
            if not pooled_raw[p]:
                entry[key] = {"note": "constant"}
                continue
            xa = np.concatenate([np.asarray(a, float) for a, _ in pooled_raw[p]])
            xb = np.concatenate([np.asarray(b, float) for _, b in pooled_raw[p]])
            pooled = round(float(spearmanr(xa, xb).correlation), 4)
            da = detrended_pooled([a for a, _ in pooled_raw[p]])
            db = detrended_pooled([b for _, b in pooled_raw[p]])
            detr = round(float(spearmanr(da, db).correlation), 4)
            entry[key] = {"pooled_rho": pooled, "detrended_rho": detr,
                          "per_seed": per_seed[p],
                          "f1_raw": bool(abs(pooled) > 0.9),
                          "f1_detrended": bool(abs(detr) > 0.9)}
        corr_report[arm] = entry

    # ---- 主判据：A4 vs 各单臂（配对种子 bootstrap）
    main = {}
    for arm in ["cka", "spectral", "beta"]:
        F4 = np.array([x["F"] for x in table["triplet"]["per_seed"]])
        Fx = np.array([x["F"] for x in table[arm]["per_seed"]])
        diff = Fx - F4
        lo, hi = bootstrap_ci(diff)
        main[arm] = {
            "delta_mean": round(float(diff.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_0": bool(lo > 0 or hi < 0),
            "wins_all_seeds": bool(all(d > 0 for d in diff)),
            "delta_ge_5pp": bool(abs(diff.mean()) >= 0.05),
        }
    # 次判据：A4 vs A0（门控 vs 不门控）
    F4 = np.array([x["F"] for x in table["triplet"]["per_seed"]])
    F0 = np.array([x["F"] for x in table["baseline"]["per_seed"]])
    diff = F0 - F4
    lo, hi = bootstrap_ci(diff)
    main["baseline"] = {
        "delta_mean": round(float(diff.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci_excludes_0": bool(lo > 0 or hi < 0),
        "wins_all_seeds": bool(all(d > 0 for d in diff)),
        "delta_ge_5pp": bool(abs(diff.mean()) >= 0.05),
    }

    # ---- triplet vs spectral 动作序列是否完全一致
    ident = {}
    for r in runs:
        a4 = next(x for x in r["arms"] if x["arm"] == "triplet")
        a2 = next(x for x in r["arms"] if x["arm"] == "spectral")
        acts4 = [x["action"] for x in a4["records"]]
        acts2 = [x["action"] for x in a2["records"]]
        ident[r["config"]["seed"]] = acts4 == acts2

    # ---- 失败条件判读
    verdicts = {
        "F1": {
            "triggered_raw": any(
                any(e["f1_raw"] for e in corr_report[arm].values()
                    if isinstance(e, dict) and "f1_raw" in e)
                for arm in arms),
            "triggered_detrended": any(
                any(e["f1_detrended"] for e in corr_report[arm].values()
                    if isinstance(e, dict) and "f1_detrended" in e)
                for arm in arms),
            "max_abs_pooled": max(
                abs(e["pooled_rho"]) for arm in arms
                for e in corr_report[arm].values()
                if isinstance(e, dict) and "pooled_rho" in e),
        },
        "F2": {
            "triggered": any(t["switch_max"] > 0.3 for t in table.values()),
            "switch_mean_by_arm": {a: t["switch_mean"] for a, t in table.items()},
            "switch_max_by_arm": {a: t["switch_max"] for a, t in table.items()},
        },
        "F5": {
            "triggered": any(t["cost_mean"] > 0.2 for t in table.values()),
            "cost_mean_by_arm": {a: t["cost_mean"] for a, t in table.items()},
        },
        "F3": {
            "note": "主判据判读见 main_criterion；A1/A3 从未告警（triplet=spectral 动作全同）"
                    "→ 对比实际只在 A4 vs A2 vs A0 之间",
            "triplet_equals_spectral_actions_all_seeds": all(ident.values()),
            "cka_beta_never_alarm": all(
                table[a]["n_isolate_total"] == 0 for a in ["cka", "beta"]),
        },
    }

    out = {"seeds": seeds, "phase_config": runs[0]["config"],
           "table": table, "correlations": corr_report,
           "main_criterion": main, "verdicts": verdicts}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- markdown 表
    L = []
    L.append("# Phase 1 pilot 聚合结果（T=10 × 5 臂 × 5 种子，width 64 校准版）")
    L.append("")
    L.append(f"- seeds: {seeds} ｜ 配置: {json.dumps({k: v for k, v in runs[0]['config'].items() if k in ('phase','T','width','steps','k','drift','noise','anchor_n','slots','tau_hi','band')}, ensure_ascii=False)}")
    L.append(f"- 复现: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase pilot --seed 0..4 --out results/pilot-metrics.json --log results/pilot-v2-seed<N>.log`")
    L.append(f"- 分析: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py`")
    L.append("")
    L.append("## 每臂汇总表")
    L.append("")
    L.append("| 臂 | F̄ ± sd | 均 acc | switch(mean/max) | isolate 总数(forced) | cost_ratio(mean) |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        t = table[arm]
        L.append(f"| {arm} | {t['F_mean']:.4f} ± {t['F_sd']:.4f} | {t['acc_mean']:.4f} "
                 f"| {t['switch_mean']:.2f}/{t['switch_max']:.2f} "
                 f"| {t['n_isolate_total']}({t['n_forced_total']}) "
                 f"| {t['cost_mean']:.3f} |")
    L.append("")
    L.append("## F1 退化检查：三量两两 Spearman（跨种子汇总，原始 / 去趋势）")
    L.append("")
    L.append("| 臂 | drift~bound | drift~beta | bound~beta |")
    L.append("|---|---|---|---|")
    for arm in arms:
        cells = []
        for p in [("drift", "bound"), ("drift", "beta"), ("bound", "beta")]:
            e = corr_report[arm][f"{p[0]}~{p[1]}"]
            if "pooled_rho" in e:
                cells.append(f"ρ={e['pooled_rho']} / 去趋势 {e['detrended_rho']}")
            else:
                cells.append("constant")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 主判据：F(triplet) vs 单臂（配对种子，bootstrap 95% CI）")
    L.append("")
    L.append("| 对手 | ΔF̄ (对手−triplet) | CI95 | 不含0 | 全部种子同向 | ≥5pp |")
    L.append("|---|---|---|---|---|---|")
    for arm, m in main.items():
        L.append(f"| {arm} | {m['delta_mean']:.4f} "
                 f"| [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] "
                 f"| {'是' if m['ci_excludes_0'] else '否'} "
                 f"| {'是' if m['wins_all_seeds'] else '否'} "
                 f"| {'是' if m['delta_ge_5pp'] else '否'} |")
    L.append("")
    L.append("## 失败条件判读（预注册 §7）")
    L.append("")
    v = verdicts
    L.append(f"- F1 退化：原始口径 max|ρ|={v['F1']['max_abs_pooled']}"
             f"（触发={v['F1']['triggered_raw']}）；去趋势口径触发={v['F1']['triggered_detrended']}")
    L.append(f"- F2 抖动：触发={v['F2']['triggered']}；各臂 switch(mean/max)="
             f"{ {a: (table[a]['switch_mean'], table[a]['switch_max']) for a in arms} }")
    L.append(f"- F5 成本：触发={v['F5']['triggered']}（阈值 0.2）；各臂 cost="
             f"{v['F5']['cost_mean_by_arm']}")
    L.append(f"- F3/结构观察：triplet 与 spectral 动作序列全部种子一致={v['F3']['triplet_equals_spectral_actions_all_seeds']}；"
             f"cka/beta 从未告警={v['F3']['cka_beta_never_alarm']}")
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(json.dumps({"F_mean": {a: table[a]["F_mean"] for a in arms},
                      "verdicts": verdicts, "main": main}, ensure_ascii=False))


if __name__ == "__main__":
    main()
