"""F2 修复第二轮（experiment-redesign-2）网格驱动：候选④集 × 5 种子。

预注册网格（notes/DECISIONS.md 2026-10-23 条，跑前冻结；参数不得在本单元内调整）：
8 个唯一规则单元 × 种子 0..4 × 5 臂 = 40 runs，pilot T=10 width64 校准版。
可恢复：metrics JSON 里已有同 (规则, 种子) 的 run 自动跳过（断点续跑）。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/run_f2fix2.py \
    [--out results/pilot-f2fix2-metrics.json] [--seeds 0,1,2,3,4] [--only TAG ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ENV = dict(os.environ, PYTHONUTF8="1")

# 预注册单元（tag 同时是日志文件名的一部分；规则字段全部冻结）
UNITS = [
    dict(tag="ctrl", f2_fix="none", persist=1, tau_hi=2.0, band=0.3,
         desc="同批对照：原规则（round-1 none/p1/tau2/b0.3 同批重跑+确定性复证）"),
    dict(tag="p2", f2_fix="none", persist=2, tau_hi=2.0, band=0.3,
         desc="家族：persist k=2（round-1 同批重跑）"),
    dict(tag="p3", f2_fix="none", persist=3, tau_hi=2.0, band=0.3,
         desc="家族：persist k=3（round-1 同批重跑）"),
    dict(tag="band", f2_fix="none", persist=1, tau_hi=2.0, band=0.5,
         desc="④a' 加宽 hysteresis band 0.3→0.5"),
    dict(tag="asym", f2_fix="asym", persist=1, tau_hi=2.0, band=0.3,
         k_in=2, k_out=3,
         desc="④a 非对称持续期：进 isolate 连续 2 告警、退连续 3 不告警"),
    dict(tag="robust", f2_fix="robust", persist=1, tau_hi=2.0, band=0.3,
         desc="④b 稳健 z（median/MAD）"),
    dict(tag="ewma", f2_fix="ewma", persist=1, tau_hi=2.0, band=0.3,
         ewma_alpha=0.3,
         desc="④b' EWMA z（alpha=0.3，隐式去趋势）"),
    dict(tag="dtb", f2_fix="detrend-bound", persist=1, tau_hi=2.0, band=0.3,
         desc="④c 混合：仅 bound 通道去趋势，drift/beta 保持原始 z"),
]


def rule_key_of(cfg):
    return (cfg.get("f2_fix"), int(cfg.get("persist", 1)),
            float(cfg.get("tau_hi", 2.0)), float(cfg.get("band", 0.3)),
            cfg.get("k_in"), cfg.get("k_out"), cfg.get("ewma_alpha"))


def done_keys(out_path):
    if not Path(out_path).exists():
        return set()
    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {(rule_key_of(r["config"]), r["config"]["seed"]) for r in data["runs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot-f2fix2-metrics.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定 tag（调试用；默认全网格）")
    ap.add_argument("--log-prefix", default="results/pilot-f2fix2-seed",
                    help="日志路径前缀（调试跑改到 scratch，避免污染正式日志）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    units = [u for u in UNITS if args.only is None or u["tag"] in args.only]

    done = done_keys(args.out)
    planned, skipped = [], 0
    for u in units:
        for seed in seeds:
            key = ((u["f2_fix"], u["persist"], u["tau_hi"], u["band"],
                    u.get("k_in"), u.get("k_out"), u.get("ewma_alpha")), seed)
            if key in done:
                skipped += 1
            else:
                planned.append((u, seed))
    print(f"grid: {len(units)} units x {len(seeds)} seeds; "
          f"to-run={len(planned)} skip(done)={skipped}", flush=True)

    t0 = time.time()
    for n, (u, seed) in enumerate(planned, 1):
        log = f"{args.log_prefix}{seed}-{u['tag']}.log"
        cmd = [PY, "experiments/train.py", "--phase", "pilot",
               "--seed", str(seed),
               "--f2-fix", u["f2_fix"], "--persist", str(u["persist"]),
               "--tau-hi", str(u["tau_hi"]), "--band", str(u["band"]),
               "--out", args.out, "--log", log]
        if u["f2_fix"] == "asym":
            cmd += ["--k-in", str(u["k_in"]), "--k-out", str(u["k_out"])]
        if u["f2_fix"] == "ewma":
            cmd += ["--ewma-alpha", str(u["ewma_alpha"])]
        print(f"[{n}/{len(planned)}] {u['tag']} seed{seed} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", env=ENV)
        if r.returncode != 0:
            print(f"FAILED {u['tag']} seed{seed} rc={r.returncode}", flush=True)
            print(r.stdout[-2000:], flush=True)
            print(r.stderr[-2000:], flush=True)
            sys.exit(1)
    print(f"done: {len(planned)} runs in {time.time() - t0:.1f}s "
          f"(skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
