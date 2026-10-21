# 状态

- 阶段：**Phase 2 完成（Split-CIFAR10，5 任务×2 类 × 3 种子，width 256）**——
  F1 存活（max|ρ|=0.60，去趋势口径一臂 −0.88 贴线）；**次判据①（门控有效性）在图像上不成立**
  （合成 5/5 一致 −41% → 图像 1 胜 1 负 1 平不显著，spectral 均 F 反超 baseline）；主判据未达
  （F3 路径延续，triplet≡beta 动作 3/3 全同）；F2/F5 触发（F5 稳态 ~11× 训练耗时）
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
- `results/cifar-metrics.json`——**Phase 2 正式 3 种子（Split-CIFAR10，runs=3）**
- `results/cifar-seed{0..2}.log`——每种子完整日志
- `results/cifar-summary.md` + `results/cifar-summary.json`——**Phase 2 主表与 F1-F5 判读**
- `results/cifar-probe.{json,log}`——steps=3 管线探针（耗时外推 + 显存验证，runs=1）
- `data/cifar10/`——CIFAR-10 解压数据（HF 镜像预置，gitignored）

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

## Phase 2 关键读数（正式结论域，3 种子，Split-CIFAR10）

| 臂 | F̄±sd | acc | switch(mean/max) | isolate 总数 | cost_ratio |
|---|---|---|---|---|---|
| baseline | 0.1121±0.0426 | 0.6831 | 0/0 | 0 | 21.98* |
| cka | 0.1121±0.0426 | 0.6831 | 0/0 | 0（从不告警） | 11.17 |
| spectral | 0.1190±0.0303 | 0.6756 | 0.25/0.50 | 2 | 11.19 |
| beta | 0.0991±0.0106 | 0.6957 | 0.33/0.50 | 2 | 11.57 |
| triplet | 0.0991±0.0106 | 0.6957 | 0.33/0.50 | 2 | 11.31 |

* baseline 臂 cost 含首次 cuSOLVER/分配器开销（3 种子一致 ~22×，疑似预热未覆盖的首次调用
  开销，根因未定位——pilot 规模下同效应不可见）；稳态用非 baseline 臂 ~11×。F5 判读不受影响
  （11× 或 22× 均远超 0.2 阈值）。

- **有效性前提成立**：baseline F̄=0.1121（种子间 0.0698-0.1550，非零对比区），acc 0.63-0.73
  ——度量路径有效，Phase 2 结论域成立。
- **F1 存活（余量变薄）**：max|ρ| 原始 0.6014（bound~beta，beta/triplet 臂）；去趋势最大
  −0.8811（drift~beta，n=12 小样本噪声大）——未越 0.9 线，三量在图像上仍可分离，H 未驳倒。
- **次判据①不跨分布（核心负结果）**：triplet vs baseline ΔF̄=+0.0129（CI95 [−0.0234, +0.0622]
  含 0；1 胜 1 负 1 平）；spectral 均 F 0.1190 > baseline 0.1121（门控反而略差）；seed1
  beta/triplet −40% 获益（0.1550→0.0928）但 seed2 +34% 反噬（0.0698→0.0932）——合成上 5/5
  种子一致 −41% 的显著性在图像分布上不保持，**门控有效性对数据分布敏感**。
- **主判据未达（F3 路径延续）**：triplet 与 beta 动作序列 3/3 种子全同（ΔF=0）；vs spectral
  Δ=0.0199（CI 含 0）；vs cka/baseline Δ≈0.013（CI 含 0）。三元组在图像上同样无决策增量。
- **F2 触发（机制与 pilot 不同）**：beta/triplet switch 0.33/0.5、spectral 0.25/0.5 > 0.3——
  但 pilot 是趋势性逐任务振荡，这里是单次尖峰触发后回 merge 的机械性 switch；T=5 下指标
  粒度粗（1 次隔离即 0.5），与 pilot 的 hysteresis 失效不同性质。
- **F5 严重触发**：稳态 cost_ratio ≈ 11×（pilot 合成 0.35×）——P=3.4M 是合成 31K 的 110×，
  SVD/J 成本主导；首臂 ~22×（首次调用开销）。论文必须报告（taste 雷②）。
- **触发结构（新事实）**：drift 通道 0/27 决策点告警（z 恒 ≤0——MLP trunk 跨任务表征随序列
  训练变稳，与合成旋转冲突结构相反）；6 次告警全部来自 task3/4 的 bound 或 beta 尖峰
  （z 2.5-7.9）；隔离收益取决于隔离槽训练充分度（100 步×2000 样本在图像上欠拟合 →
  seed2 回滚+重训反噬 +34%）。

## 难度校准 v2（已拍板）
width 256 时遗忘被容量压到 0.003~0.005（近饱和，spectral 从不告警，对比区不存在）→
校准扫描定 **pilot = T=10 × width=64 × steps=100 × drift=0.5 × noise=1.0 × k=10**
（与 smoke 同难度协议，仅 T/步数放量）；width 256 为否决配置（见 calibration-sweep.txt）。

## 复现入口
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase pilot --seed 0..4 \
      --out results/pilot-metrics.json --log results/pilot-v2-seed<N>.log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase cifar --seed 0,1,2 \
      --out results/cifar-metrics.json --log results/cifar-seed<N>.log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_pilot.py \
      --in results/cifar-metrics.json --md results/cifar-summary.md \
      --json results/cifar-summary.json   # Phase 2 聚合+判读

## git
- 本地仓库 main 分支，今晚 7 commits（建档 785ea69 / smoke 1f2d427 / 收工 ed238b7 /
  校准 v2 8573eb5 / pilot 结果+文档 119d61a / STATUS SHA 修正 ad57ffc / Phase 2
  结果+脚本通用化 / Phase 2 判读落档），未推送

## 下一步（next_units）
1. **F2 重设计 + pilot 重跑**（Phase 1 遗留，预注册必经步）：bound 趋势性导致 z 振荡，
   候选：对 bound 去趋势/提高 τ/动作持续期平滑；验收：switch<0.3 且 F 不劣化。~35min
2. **图像门控重设计（Phase 2 负结果后续）**：候选——τ 分布校准（只能作敏感性分析报告，
   防事后调参）/ 尖峰持续期确认（连续边界告警才切）/ 隔离槽步数随任务规模放大；验收：图像上
   门控效果符号一致且不显著劣化于 baseline。~45min
3. Phase 3：合成 T=50 长序列（F2 修复后）——"长序列才显著"预测 + 三元组增量复判。~45min

## 阻塞
- 无（F2 与图像触发惰性均为规则层问题，有明确根因假设与候选修法）
