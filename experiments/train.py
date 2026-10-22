"""训练主循环：五臂门控对比 + 三量触发计算 + 遗忘度量（pilot-design.md §1-§5）。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase smoke --seed 0 \
    --out results/metrics.json --log results/smoke-log.txt

phase: smoke（T=5 管线验证）/ pilot（T=10 正式 pilot）/ cifar（Split-CIFAR10）。
五臂同数据同初始化同协议，只差进入告警的触发通道；所有臂都计算全部三量。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks import make_task_sequence, make_split_cifar
from model import SlotsModel
from triggers import linear_cka, anchor_jacobians, spectral_bound, subspace_leakage
from gating import CHANNELS, RunningZ, TrendAdjustedZ, RobustZ, EWMAZ, decide

ARMS = {
    "baseline": [],
    "cka": ["drift"],
    "spectral": ["bound"],
    "beta": ["beta"],
    "triplet": ["drift", "bound", "beta"],
}

PHASES = {
    # 难度校准：k=4/noise=0.5 时准确率饱和、遗忘恒 0，度量路径走不到对比区；
    # 难度必须让 baseline 有非零遗忘（pilot-design.md §4）。
    "smoke": dict(T=5, d=30, k=10, train_n=600, test_n=256, anchor_n=32,
                  width=64, steps=60, batch=256, lr=1e-3, slots=4,
                  noise=1.0, drift=0.5),
    # 难度校准 v2（2026-10-22 夜间单元）：width 256 时 baseline 遗忘被容量压到
    # 0.003-0.005（近饱和、门控从不告警，对比区不存在）；校准扫描
    # （seed 0，T=10）：w256/drift1.0 F=0.0030；w128/drift1.0 F=0.0140；
    # w64/drift1.0 F=0.0342；w64/drift0.5 F=0.0534（acc 0.9436，非零对比区）。
    # 取 w64/drift0.5（与 smoke 同难度协议，仅 T/步数放量），width 256 记为
    # pilot-design.md §4 的否决配置（见 DECISIONS.md）。
    "pilot": dict(T=10, d=30, k=10, train_n=2000, test_n=500, anchor_n=64,
                  width=64, steps=100, batch=256, lr=1e-3, slots=4,
                  noise=1.0, drift=0.5),
    "cifar": dict(T=5, d=3072, k=2, train_n=2000, test_n=500, anchor_n=64,
                  width=256, steps=100, batch=256, lr=1e-3, slots=4),
}


def log_write(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate(model, task, slot, head, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(task["Xte"].to(device), slot, head).argmax(1).cpu()
    acc = float((pred == task["yte"]).float().mean())
    if was_training:
        model.train()
    return acc


def train_steps(model, group, task, slot, head, device, steps, batch, lr):
    opt = torch.optim.Adam(group, lr=lr)
    X, y = task["Xtr"].to(device), task["ytr"].to(device)
    n = X.shape[0]
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        opt.zero_grad()
        loss = F.cross_entropy(model(X[idx], slot, head), y[idx])
        loss.backward()
        opt.step()
    return float(loss.detach())


def run_arm(arm_name, tasks, cfg, seed, device, tau_hi, band, log,
            f2_fix="none", persist=1, k_in=None, k_out=None, ewma_alpha=0.3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SlotsModel(cfg["d"], cfg["width"], cfg["k"], cfg["T"], cfg["slots"]).to(device)
    active = ARMS[arm_name]
    assignments = [0] * cfg["T"]
    free_slots = list(range(1, cfg["slots"]))
    # F2 修复候选（预注册见 gating.py 模块 docstring / notes/DECISIONS.md）：
    # none/persist/asym → RunningZ；detrend → 全通道 TrendAdjustedZ；
    # detrend-bound → 仅 bound 去趋势（drift/beta 保持原始 z）；robust → 中位数/MAD；
    # ewma → EWMA z（alpha 可调）。
    if f2_fix == "detrend-bound":
        zstats = {"drift": RunningZ(), "bound": TrendAdjustedZ(), "beta": RunningZ()}
    elif f2_fix == "detrend":
        zstats = {c: TrendAdjustedZ() for c in CHANNELS}
    elif f2_fix == "robust":
        zstats = {c: RobustZ() for c in CHANNELS}
    elif f2_fix == "ewma":
        zstats = {c: EWMAZ(alpha=ewma_alpha) for c in CHANNELS}
    else:
        zstats = {c: RunningZ() for c in CHANNELS}
    rule = {"f2_fix": f2_fix, "tau_hi": tau_hi, "band": band, "persist": persist}
    if f2_fix == "asym":
        rule["k_in"], rule["k_out"] = k_in, k_out
    if f2_fix == "ewma":
        rule["ewma_alpha"] = ewma_alpha
    deltas = []
    feats_cache = {}
    eval_hist = []
    records = []
    prev_action = "merge"
    gate_state = {}
    names = [n for n, _ in model.named_parameters()]
    t_jac = t_svd = t_cka = t_train = t_iso = 0.0

    def flat():
        return torch.cat([p.detach().reshape(-1) for _, p in model.named_parameters()])

    for i in range(cfg["T"]):
        task = tasks[i]
        # 决策前快照（回滚用）+ requires_grad 参数副本（J 用）
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        params_pre = {n: p.detach().clone().requires_grad_(True)
                      for n, p in model.named_parameters()}

        # 锚任务梯度矩阵 J_r（在任务 i 训练前状态）
        Js = {}
        if i >= 1:
            tj = time.perf_counter()
            Js = anchor_jacobians(model, params_pre, tasks[:i], assignments[:i], device)
            t_jac += time.perf_counter() - tj

        # 试训练：默认并入 trunk0
        tt = time.perf_counter()
        train_steps(model, model.train_params(0, i), task, 0, i, device,
                    cfg["steps"], cfg["batch"], cfg["lr"])
        t_train += time.perf_counter() - tt

        dW = flat() - torch.cat([snap[n].reshape(-1) for n in names])

        # 三量触发计算（任务 2 起，0-indexed i>=1）
        raws = {c: None for c in CHANNELS}
        zs = {c: 0.0 for c in CHANNELS}
        if i >= 1:
            ts = time.perf_counter()
            raws["bound"] = spectral_bound(dW, Js)
            t_svd += time.perf_counter() - ts
            tc = time.perf_counter()
            with torch.no_grad():
                cur_feat = model.trunks[0](tasks[i - 1]["Xan"].to(device))
            raws["drift"] = 1.0 - linear_cka(feats_cache[i - 1][i - 1], cur_feat)
            raws["beta"] = subspace_leakage([deltas[i - 1], dW], Js)
            t_cka += time.perf_counter() - tc

        # 门控决策（第 3 个任务起，0-indexed i>=2；baseline 臂永不门控）
        action, max_z, override, forced = "merge", 0.0, False, False
        if i >= 2 and active:
            for c in CHANNELS:
                zs[c] = zstats[c].z(i, raws[c])
            action, max_z, override, gate_state = decide(
                zs, active, prev_action, tau_hi, band, persist,
                k_in=k_in if f2_fix == "asym" else None,
                k_out=k_out if f2_fix == "asym" else None,
                state=gate_state)

        if action == "isolate":
            if free_slots:
                slot = free_slots.pop(0)
                model.load_state_dict(snap)  # 回滚并入试训练
                assignments[i] = slot
                ti = time.perf_counter()
                train_steps(model, model.train_params(slot, i), task, slot, i, device,
                            cfg["steps"], cfg["batch"], cfg["lr"])
                t_iso += time.perf_counter() - ti
            else:
                action, forced = "merge", True  # 槽位耗尽 -> 强制并入

        # z 统计只观察先于当前任务的样本（决策后观察）
        if i >= 1:
            for c in CHANNELS:
                zstats[c].observe(i, raws[c])
        deltas.append(dW)

        # 全部已见任务评估（遗忘度量）
        accs = {str(j): evaluate(model, tasks[j], assignments[j], j, device)
                for j in range(i + 1)}
        eval_hist.append({int(k): v for k, v in accs.items()})
        # 任务边界缓存 trunk0 特征（下一任务 CKA 用）
        with torch.no_grad():
            feats_cache[i] = {r: model.trunks[0](tasks[r]["Xan"].to(device))
                              for r in range(i + 1)}

        records.append({
            "task": i, "action": action, "forced": forced,
            "hysteresis_override": override,
            "raws": {c: (None if raws[c] is None else round(float(raws[c]), 6))
                     for c in CHANNELS},
            "zs": {c: round(zs[c], 4) for c in CHANNELS},
            "max_z": round(max_z, 4), "slot": assignments[i],
            "evals": {k: round(v, 4) for k, v in accs.items()},
        })
        log(f"  [{arm_name}] task {i}: action={action}"
            + (" (forced)" if forced else "")
            + (" (hyst)" if override else "")
            + f" slot={assignments[i]}"
            + f" drift={raws['drift']} bound={raws['bound']} beta={raws['beta']}"
            + f" accs={ {int(k): round(v, 3) for k, v in accs.items()} }")
        prev_action = action

    # ---- 序列级汇总
    final = eval_hist[-1]
    best = {}
    for h in eval_hist:
        for j, a in h.items():
            best[j] = max(best.get(j, 0.0), a)
    forgetting = {j: best[j] - final[j] for j in final}
    actions = [r["action"] for r in records]
    switches = sum(1 for k in range(1, len(actions)) if actions[k] != actions[k - 1])

    # 三量两两 Spearman（退化检查 F1；任务 1 起有 raws）
    series = {c: [records[i]["raws"][c] for i in range(1, len(records))]
              for c in CHANNELS}
    corr = {}
    for a in range(len(CHANNELS)):
        for b in range(a + 1, len(CHANNELS)):
            ca, cb = CHANNELS[a], CHANNELS[b]
            xa, xb = series[ca], series[cb]
            if len(set(xa)) <= 1 or len(set(xb)) <= 1:
                corr[f"{ca}~{cb}"] = {"rho": 1.0, "constant": True}
            else:
                rho = float(spearmanr(xa, xb).correlation)
                corr[f"{ca}~{cb}"] = {"rho": round(rho, 4), "constant": False}

    cost_ratio = (t_jac + t_svd + t_cka) / max(t_train, 1e-9)
    summary = {
        "rule": rule,
        "mean_forgetting": round(float(np.mean(list(forgetting.values()))), 4),
        "mean_final_acc": round(float(np.mean(list(final.values()))), 4),
        "per_task_forgetting": {str(j): round(v, 4) for j, v in forgetting.items()},
        "n_merge": actions.count("merge"),
        "n_isolate": actions.count("isolate"),
        "n_forced": sum(1 for r in records if r["forced"]),
        "action_switch_rate": round(switches / max(len(actions) - 1, 1), 4),
        "correlations": corr,
        "param_count": model.param_count(),
        "slots_used": len(set(assignments)),
        "timings_sec": {"jacobian": round(t_jac, 3), "svd": round(t_svd, 3),
                        "cka": round(t_cka, 3), "train_merge": round(t_train, 3),
                        "train_isolate": round(t_iso, 3)},
        "trigger_cost_ratio": round(cost_ratio, 4),
    }
    return {"arm": arm_name, "active_channels": active,
            "records": records, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke", choices=list(PHASES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--log", default="results/smoke-log.txt")
    ap.add_argument("--tau-hi", type=float, default=2.0)
    ap.add_argument("--band", type=float, default=0.3)
    # F2 修复候选（预注册见 gating.py 模块 docstring / notes/DECISIONS.md）：
    # none=原规则（回归对照）/ detrend=全通道去趋势 z /
    # detrend-bound=仅 bound 通道去趋势（drift/beta 保持原始 z）/
    # persist=动作持续期平滑（连续 persist 个边界确认才切换）/
    # asym=非对称持续期（进 isolate 需连续 k-in 告警、退需连续 k-out 不告警）/
    # robust=中位数/MAD 稳健 z / ewma=EWMA z（--ewma-alpha）。
    # tau 提高用 --tau-hi 传（阈值类修法只作敏感性分析，见 workflow.json 约定）。
    ap.add_argument("--f2-fix", default="none",
                    choices=["none", "detrend", "detrend-bound", "persist",
                             "asym", "robust", "ewma"])
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--k-in", type=int, default=2,
                    help="asym：进 isolate 需连续告警边界数（预注册=2）")
    ap.add_argument("--k-out", type=int, default=3,
                    help="asym：退 isolate 需连续不告警边界数（预注册=3）")
    ap.add_argument("--ewma-alpha", type=float, default=0.3,
                    help="ewma：指数加权 alpha（预注册=0.3）")
    ap.add_argument("--device", default="auto")
    # 难度校准覆盖（pilot-design.md §4 要求 baseline 遗忘非零才有效度；
    # 宽度提升会压掉遗忘，这些覆盖只用于校准扫描，校准结果拍板后烘焙进 PHASES）
    for flag, key, typ in [("--width", "width", int), ("--drift", "drift", float),
                           ("--noise", "noise", float), ("--k", "k", int),
                           ("--steps", "steps", int)]:
        ap.add_argument(flag, type=typ, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    cfg = dict(PHASES[args.phase])
    # 应用难度校准覆盖（None 保持 phase 默认值）
    for key in ["width", "drift", "noise", "k", "steps"]:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = int(v) if key in ("k", "steps", "width") else float(v)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"# run phase={args.phase} seed={args.seed} device={device} torch={torch.__version__}")
    log(f"# rule: f2_fix={args.f2_fix} persist={args.persist} "
        f"tau_hi={args.tau_hi} band={args.band}"
        + (f" k_in={args.k_in} k_out={args.k_out}" if args.f2_fix == "asym" else "")
        + (f" ewma_alpha={args.ewma_alpha}" if args.f2_fix == "ewma" else ""))
    log(f"# config={json.dumps(cfg, ensure_ascii=False)}")

    if args.phase == "cifar":
        tasks = make_split_cifar(args.seed, T=cfg["T"], train_n=cfg["train_n"],
                                 test_n=cfg["test_n"], anchor_n=cfg["anchor_n"])
    else:
        tasks = make_task_sequence(args.seed, cfg["T"], cfg["d"], cfg["k"],
                                   cfg["train_n"], cfg["test_n"], cfg["anchor_n"],
                                   drift=cfg.get("drift", 0.3),
                                   noise=cfg.get("noise", 0.5))

    # 预热：按本 phase 精确形状（参数量/锚集/批量）摊薄一次性 CUDA 初始化/
    # 编译开销，保证 F5 成本比（trigger_cost_ratio）是稳态值、臂间可比
    warm_model = SlotsModel(cfg["d"], cfg["width"], cfg["k"],
                            cfg["T"], cfg["slots"]).to(device)
    warm_tasks = make_task_sequence(args.seed + 9999, 2, cfg["d"], cfg["k"],
                                    max(cfg["batch"], 64) + 16, 32, cfg["anchor_n"],
                                    drift=cfg.get("drift", 0.3),
                                    noise=cfg.get("noise", 0.5))
    warm_params = {n: p.detach().clone().requires_grad_(True)
                   for n, p in warm_model.named_parameters()}
    _Js = anchor_jacobians(warm_model, warm_params, warm_tasks[:1], [0], device)
    _dW = torch.randn(warm_model.param_count(), device=device)
    spectral_bound(_dW, _Js)
    subspace_leakage([_dW, _dW], _Js)
    linear_cka(torch.randn(cfg["anchor_n"], cfg["width"], device=device),
               torch.randn(cfg["anchor_n"], cfg["width"], device=device))
    train_steps(warm_model, warm_model.train_params(0, 1), warm_tasks[1], 0, 1,
                device, 3, cfg["batch"], 1e-3)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    arms_out = []
    for arm in ARMS:
        log(f"== arm {arm} (active={ARMS[arm]}) ==")
        arms_out.append(run_arm(arm, tasks, cfg, args.seed, device,
                                args.tau_hi, args.band, log,
                                args.f2_fix, args.persist,
                                args.k_in, args.k_out, args.ewma_alpha))
    total = time.perf_counter() - t0

    log("== summary ==")
    for a in arms_out:
        s = a["summary"]
        log(f"  {a['arm']:>10}: F={s['mean_forgetting']:.4f} "
            f"acc={s['mean_final_acc']:.4f} merge={s['n_merge']} "
            f"iso={s['n_isolate']} forced={s['n_forced']} "
            f"switch={s['action_switch_rate']:.2f} "
            f"cost_ratio={s['trigger_cost_ratio']:.3f} params={s['param_count']}")

    run_record = {
        "run_id": f"{args.phase}-seed{args.seed}-{int(time.time())}",
        "config": {"phase": args.phase, "seed": args.seed, "tau_hi": args.tau_hi,
                   "band": args.band, "device": device, "torch": torch.__version__,
                   "f2_fix": args.f2_fix, "persist": args.persist,
                   **cfg},
        "wall_sec": round(total, 2),
        "arms": arms_out,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("runs", []).append(run_record)
    else:
        data = {"runs": [run_record]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    log(f"# wrote {out_path} (runs={len(data['runs'])})")

    log_write(args.log, lines)
    print(f"log -> {args.log}", flush=True)


if __name__ == "__main__":
    main()
