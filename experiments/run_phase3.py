"""Phase 3 长序列终判（experiment-long）网格驱动：asym 规则 × 5 种子 × 5 臂。

预注册协议（notes/DECISIONS.md 2026-10-23 拍板条 + workflow.json
minimal_discriminative_experiment.decisive_step，跑前冻结；参数不得在本单元内调整）：
- 门控规则 = asym（f2_fix=asym, k_in=2, k_out=3, persist=1, tau_hi=2.0, band=0.3）
  ——预注册候选④a 参数；F2 gate 形式上未过（T=10 同批 seed4 配对 ΔF=+0.0110>0.005，
  switch_max 0.22 与 F̄ 0.0304 已过）→ 按组合拍板以「限制报告」携带进终判。
- 难度协议 = 合成 pilot 校准版（width 64 / steps 100 / drift 0.5 / noise 1.0 /
  k 10 / anchor 64 / train_n 2000），仅 T=50（train.py --T 覆盖，PHASES 无 T=50 入口）。
- 网格：T=50 × 5 臂（baseline/cka/spectral/beta/triplet）× 5 种子 = 25 runs
  （train.py 单次调用跑完 5 臂）。
可恢复：metrics JSON 里已有同 (规则含 T, 种子) 的 run 自动跳过（断点续跑）。

用法:
  PYTHONUTF8=1 G:/ai/python11/python.exe experiments/run_phase3.py \
    [--out results/phase3-metrics.json] [--seeds 0,1,2,3,4] [--T 50]
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

# 预注册规则单元（唯一；tag 同时是日志文件名的一部分。规则字段全部冻结）
UNIT = dict(tag="asym", f2_fix="asym", persist=1, tau_hi=2.0, band=0.3,
            k_in=2, k_out=3,
            desc="④a 非对称持续期：进 isolate 连续 2 告警、退连续 3 不告警"
                 "（Phase 3 拍板规则；F2 限制报告携带）")


def rule_key_of(cfg):
    """单元签名：全部行为轴 + T（轮次 1/2 教训：标签缺轴会静默合并单元）。"""
    return (cfg.get("f2_fix"), int(cfg.get("persist", 1)),
            float(cfg.get("tau_hi", 2.0)), float(cfg.get("band", 0.3)),
            cfg.get("k_in"), cfg.get("k_out"), cfg.get("ewma_alpha"),
            int(cfg.get("T", -1)))


def done_keys(out_path):
    if not Path(out_path).exists():
        return set()
    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {(rule_key_of(r["config"]), r["config"]["seed"]) for r in data["runs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3-metrics.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--T", type=int, default=50, help="序列长度（预注册=50）")
    ap.add_argument("--log-prefix", default="results/phase3-seed",
                    help="日志路径前缀（调试跑改到 scratch，避免污染正式日志）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    u = UNIT

    done = done_keys(args.out)
    planned, skipped = [], 0
    for seed in seeds:
        key = ((u["f2_fix"], u["persist"], u["tau_hi"], u["band"],
                u["k_in"], u["k_out"], None, args.T), seed)
        if key in done:
            skipped += 1
        else:
            planned.append(seed)
    print(f"phase3 grid: rule={u['tag']} T={args.T} x {len(seeds)} seeds x 5 arms; "
          f"to-run={len(planned)} skip(done)={skipped}", flush=True)

    t0 = time.time()
    for n, seed in enumerate(planned, 1):
        log = f"{args.log_prefix}{seed}.log"
        cmd = [PY, "experiments/train.py", "--phase", "pilot",
               "--seed", str(seed),
               "--T", str(args.T),
               "--f2-fix", u["f2_fix"], "--persist", str(u["persist"]),
               "--tau-hi", str(u["tau_hi"]), "--band", str(u["band"]),
               "--k-in", str(u["k_in"]), "--k-out", str(u["k_out"]),
               "--out", args.out, "--log", log]
        print(f"[{n}/{len(planned)}] {u['tag']} T={args.T} seed{seed} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", env=ENV)
        if r.returncode != 0:
            print(f"FAILED {u['tag']} T={args.T} seed{seed} rc={r.returncode}", flush=True)
            print(r.stdout[-2000:], flush=True)
            print(r.stderr[-2000:], flush=True)
            sys.exit(1)
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("# wrote")]
        print(f"  ok ({tail[-1] if tail else 'see log'})", flush=True)
    print(f"done: {len(planned)} runs in {time.time() - t0:.1f}s "
          f"(skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
