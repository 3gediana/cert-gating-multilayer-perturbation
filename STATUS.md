# 状态

- 阶段：**Phase 1 完成（T=10 五臂 × 5 种子初判，校准版 width 64）**——
  主判据未达（F3 路径）；F1 未触发；F2/F5 触发（工程与规则层，记 FAILURE-PATTERNS）
- 最近更新：2026-10-22（夜间生产单元，autonomous-research-production 模式）

## 产物
- `experiments/pilot-design.md`——五臂协议 + 三量操作化 + 预注册判读标准 + 失败条件 F1-F5
- `experiments/{tasks,model,triggers,gating,train}.py`——训练主循环（pilot 难度已校准 v2）
- `experiments/analyze_pilot.py`——Phase 1 聚合分析（遗忘表/F1 跨种子相关/配对 bootstrap/
  F1-F5 判读；原始 + 去趋势双口径）
- `results/metrics.json`——smoke + pilot width-256 seed0（否决配置历史记录，runs=2）
- `results/pilot-metrics.json`——**正式 pilot 5 种子（width 64 校准版）**
- `results/pilot-v2-seed{0..4}.log`——每种子完整日志
- `results/pilot-summary.md` + `results/pilot-summary.json`——**主表与判读**
- `results/calibration-sweep.txt`——难度校准扫描证据（4 配置 × seed 0）

## Phase 1 关键读数（正式结论域，5 种子）

| 臂 | F̄±sd | acc | switch(mean/max) | isolate | cost_ratio |
|---|---|---|---|---|---|
| baseline | 0.0557±0.0247 | 0.9397 | 0/0 | 0 | 0.350 |
| cka | 0.0557±0.0247 | 0.9397 | 0/0 | 0（从不告警） | 0.355 |
| spectral | **0.0331±0.0173** | 0.9636 | 0.49/0.67 | 14(2 forced) | 0.351 |
| beta | 0.0529±0.0260 | 0.9426 | 0.04/0.22 | 1 | 0.361 |
| triplet | **0.0331±0.0173** | 0.9636 | 0.49/0.67 | 14(2 forced) | 0.361 |

- **F1 退化检查：未触发**。跨种子汇总（n=45/臂）max|ρ|=0.68（baseline drift~bound）；
  去趋势后全部 |ρ|<0.35——三量不退化。但原始 vs 去趋势差异大：三量随任务序号有
  趋势性共变（width 256 单种子曾见 ρ=−0.9，是 n=9 + 趋势的假象，n=45 后回落）。
- **主判据：未达**。triplet 与 spectral 动作序列 5/5 种子完全一致（Δ=0）；vs cka/beta
  Δ≈0.02 且 CI95 不含 0、5/5 同向，但 <5pp 条款。校准难度下只有 bound 通道告警
  （cka 0/50 决策从未告警，beta 仅 seed2 一次），三元组的额外通道未产生任何决策增量
  → 按 F3 预注册：pilot 尺度 H 未获支持，升级长序列 Phase 3 再判。
- **有效事实**：spectral 单臂门控使遗忘 0.0557→0.0331（相对 −41%，5/5 种子同向，
  配对 bootstrap CI [0.0158, 0.0297] 不含 0）——门控本身有效（次判据①成立），
  但三元组相对单量无增量。
- **F2 抖动：触发**。spectral/triplet switch 0.49/0.67 > 0.3——hysteresis 失效。
  根因假设：bound 随任务序号单调增长（seed0 量级 27→581），running z 均值滞后 →
  晚期持续告警、早期合并 → 振荡。下一单元按 F2 预注册"重设计后再判"。
- **F5 成本：触发**。cost_ratio ~0.35 > 0.2（SVD 主导：J 为 m×P 的 svdvals）。
  结构发现：触发成本随 T 超线性（J 覆盖全部锚任务 ~O(T²·m²P)，训练 O(T·steps)）；
  width 256 时 cost 达 1.5+。论文可扩展性必须报告（taste 雷）。

## 难度校准 v2（已拍板）
width 256 时遗忘被容量压到 0.003~0.005（近饱和，spectral 从不告警，对比区不存在）→
校准扫描定 **pilot = T=10 × width=64 × steps=100 × drift=0.5 × noise=1.0 × k=10**
（与 smoke 同难度协议，仅 T/步数放量）；width 256 为否决配置（见 calibration-sweep.txt）。

## 复现入口
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase pilot --seed 0..4 \
      --out results/pilot-metrics.json --log results/pilot-v2-seed<N>.log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py   # 聚合+判读

## git
- 本地仓库 main 分支，今晚 4+ commits（建档 785ea69 / smoke 1f2d427 / 收工 ed238b7 /
  校准 v2 8573eb5 / pilot 结果待提交），未推送

## 下一步（next_units）
1. **F2 重设计 + pilot 重跑**（预注册 F2 触发后的必经步）：bound 趋势性导致 z 振荡，
   候选：对 bound 去趋势/提高 τ/动作持续期平滑；验收：switch<0.3 且 F 不劣化。~35min
2. Phase 3：合成 T=50 长序列（F2 修复后）——"长序列才显著"预测 + 三元组增量复判。~45min
3. Phase 2：Split-CIFAR10 子集 3 种子（F1 未触发，前置已满足）。~40min

## 阻塞
- 无（F2 是规则层问题，已有明确根因假设与候选修法）
