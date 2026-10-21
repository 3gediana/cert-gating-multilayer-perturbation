# Phase 2 Split-CIFAR10 聚合结果（T=5 × 5 臂 × 3 种子）

- seeds: [0, 1, 2] ｜ 配置: {"phase": "cifar", "tau_hi": 2.0, "band": 0.3, "T": 5, "k": 2, "anchor_n": 64, "width": 256, "steps": 100, "slots": 4}
- 复现: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase cifar --seed 0,1,2 --out results/cifar-metrics.json --log results/cifar-seed<N>.log`
- 分析: `PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py`

## 每臂汇总表

| 臂 | F̄ ± sd | 均 acc | switch(mean/max) | isolate 总数(forced) | cost_ratio(mean) |
|---|---|---|---|---|---|
| baseline | 0.1121 ± 0.0426 | 0.6831 | 0.00/0.00 | 0(0) | 21.976 |
| cka | 0.1121 ± 0.0426 | 0.6831 | 0.00/0.00 | 0(0) | 11.168 |
| spectral | 0.1190 ± 0.0303 | 0.6756 | 0.25/0.50 | 2(0) | 11.192 |
| beta | 0.0991 ± 0.0106 | 0.6957 | 0.33/0.50 | 2(0) | 11.573 |
| triplet | 0.0991 ± 0.0106 | 0.6957 | 0.33/0.50 | 2(0) | 11.312 |

## F1 退化检查：三量两两 Spearman（跨种子汇总，原始 / 去趋势）

| 臂 | drift~bound | drift~beta | bound~beta |
|---|---|---|---|
| baseline | ρ=-0.4685 / 去趋势 0.2937 | ρ=-0.2867 / 去趋势 -0.4476 | ρ=0.1818 / 去趋势 0.021 |
| cka | ρ=-0.4685 / 去趋势 0.2937 | ρ=-0.2867 / 去趋势 -0.4476 | ρ=0.1818 / 去趋势 0.021 |
| spectral | ρ=-0.3497 / 去趋势 0.0699 | ρ=-0.3077 / 去趋势 -0.4545 | ρ=0.2937 / 去趋势 0.1818 |
| beta | ρ=-0.2797 / 去趋势 -0.042 | ρ=-0.4266 / 去趋势 -0.8811 | ρ=0.6014 / 去趋势 0.2797 |
| triplet | ρ=-0.2797 / 去趋势 -0.042 | ρ=-0.4266 / 去趋势 -0.8811 | ρ=0.6014 / 去趋势 0.2797 |

## 主判据：F(triplet) vs 单臂（配对种子，bootstrap 95% CI）

| 对手 | ΔF̄ (对手−triplet) | CI95 | 不含0 | 全部种子同向 | ≥5pp |
|---|---|---|---|---|---|
| cka | 0.0129 | [-0.0234, 0.0622] | 否 | 否 | 否 |
| spectral | 0.0199 | [0.0000, 0.0596] | 否 | 否 | 否 |
| beta | 0.0000 | [0.0000, 0.0000] | 否 | 否 | 否 |
| baseline | 0.0129 | [-0.0234, 0.0622] | 否 | 否 | 否 |

## 失败条件判读（预注册 §7）

- F1 退化：原始口径 max|ρ|=0.6014（触发=False）；去趋势口径触发=False
- F2 抖动：触发=True；各臂 switch(mean/max)={'baseline': (0.0, 0.0), 'cka': (0.0, 0.0), 'spectral': (0.25, 0.5), 'beta': (0.3333, 0.5), 'triplet': (0.3333, 0.5)}
- F5 成本：触发=True（阈值 0.2）；各臂 cost={'baseline': 21.9759, 'cka': 11.1683, 'spectral': 11.1921, 'beta': 11.5734, 'triplet': 11.3121}
- F3/结构观察：triplet 与 spectral 动作序列全部种子一致=False；cka/beta 从未告警=False
