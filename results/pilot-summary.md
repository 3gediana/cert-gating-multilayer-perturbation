# Phase 1 pilot 聚合结果（T=10 × 5 臂 × 5 种子，width 64 校准版）

- seeds: [0, 1, 2, 3, 4] ｜ 配置: {"phase": "pilot", "tau_hi": 2.0, "band": 0.3, "T": 10, "k": 10, "anchor_n": 64, "width": 64, "steps": 100, "slots": 4, "noise": 1.0, "drift": 0.5}
- 复现: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase pilot --seed 0..4 --out results/pilot-metrics.json --log results/pilot-v2-seed<N>.log`
- 分析: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py`

## 每臂汇总表

| 臂 | F̄ ± sd | 均 acc | switch(mean/max) | isolate 总数(forced) | cost_ratio(mean) |
|---|---|---|---|---|---|
| baseline | 0.0557 ± 0.0247 | 0.9397 | 0.00/0.00 | 0(0) | 0.350 |
| cka | 0.0557 ± 0.0247 | 0.9397 | 0.00/0.00 | 0(0) | 0.355 |
| spectral | 0.0331 ± 0.0173 | 0.9636 | 0.49/0.67 | 14(2) | 0.351 |
| beta | 0.0529 ± 0.0260 | 0.9426 | 0.04/0.22 | 1(0) | 0.361 |
| triplet | 0.0331 ± 0.0173 | 0.9636 | 0.49/0.67 | 14(2) | 0.361 |

## F1 退化检查：三量两两 Spearman（跨种子汇总，原始 / 去趋势）

| 臂 | drift~bound | drift~beta | bound~beta |
|---|---|---|---|
| baseline | ρ=-0.6659 / 去趋势 0.165 | ρ=0.2852 / 去趋势 0.2099 | ρ=-0.5755 / 去趋势 -0.23 |
| cka | ρ=-0.6659 / 去趋势 0.165 | ρ=0.2852 / 去趋势 0.2099 | ρ=-0.5755 / 去趋势 -0.23 |
| spectral | ρ=-0.4813 / 去趋势 -0.1602 | ρ=0.0339 / 去趋势 -0.3119 | ρ=-0.4491 / 去趋势 -0.0165 |
| beta | ρ=-0.6818 / 去趋势 0.0941 | ρ=0.2271 / 去趋势 0.1283 | ρ=-0.513 / 去趋势 -0.2018 |
| triplet | ρ=-0.4813 / 去趋势 -0.1602 | ρ=0.0339 / 去趋势 -0.3119 | ρ=-0.4491 / 去趋势 -0.0165 |

## 主判据：F(triplet) vs 单臂（配对种子，bootstrap 95% CI）

| 对手 | ΔF̄ (对手−triplet) | CI95 | 不含0 | 全部种子同向 | ≥5pp |
|---|---|---|---|---|---|
| cka | 0.0226 | [0.0158, 0.0297] | 是 | 是 | 否 |
| spectral | 0.0000 | [0.0000, 0.0000] | 否 | 否 | 否 |
| beta | 0.0198 | [0.0118, 0.0285] | 是 | 是 | 否 |
| baseline | 0.0226 | [0.0158, 0.0297] | 是 | 是 | 否 |

## 失败条件判读（预注册 §7）

- F1 退化：原始口径 max|ρ|=0.6818（触发=False）；去趋势口径触发=False
- F2 抖动：触发=True；各臂 switch(mean/max)={'baseline': (0.0, 0.0), 'cka': (0.0, 0.0), 'spectral': (0.4889, 0.6667), 'beta': (0.0444, 0.2222), 'triplet': (0.4889, 0.6667)}
- F5 成本：触发=True（阈值 0.2）；各臂 cost={'baseline': 0.3503, 'cka': 0.3549, 'spectral': 0.3507, 'beta': 0.3607, 'triplet': 0.3611}
- F3/结构观察：triplet 与 spectral 动作序列全部种子一致=True；cka/beta 从未告警=False
