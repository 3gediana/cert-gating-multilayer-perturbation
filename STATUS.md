# 状态

- 阶段：**experiment-long（Phase 3 终判）完成——H1 在合成尺度 falsified（F3 终判，负结果）**：T=50 × 5 臂 × 5 种子，triplet vs spectral ΔF̄=0（动作 5/5 全同）、vs cka/beta +0.0374（CI95 不含 0 但 <5pp）→ 主判据 FAIL；F1 未触发（0.7331<0.9）、F2 T=50 未触发（0.04；T=10 gate 未过状态以限制报告并列）、F5 严重触发（cost_ratio 3.10，成本曲线已附）。详见「Phase 3 终判读数」节。
- 最近更新：2026-10-23 09:2x（experiment-long 单元：--T 管线步骤 + 协议冻结 commit 36be619 → 25 runs（804s）+ T=25 成本探针 + seed0 重跑逐位复证 → analyze_phase3 判读落档）
- 收尾恢复点（2026-10-23 07:55，wrap 收线）：experiment-redesign-2 **40/40 runs 完成（736.9s，skipped 0）+ analyze_f2fix2 判读落盘 results/pilot-f2fix2-summary.{md,json}**。判读：7/7 候选主/次口径全 FAIL（预注册 gate 三条件）；**asym（④a 非对称 i2o3）最近线**——switch 0.22/0.22 全种子一致、F̄ 0.0304≤0.0331、4/5 种子配对改善（Δ=-0.007~-0.010），仅 seed4 ΔF=+0.0110>PAIR_TOL 0.005 触发 paired_not_worse=false；对照 none/p1/tau2/b0.3 逐位复证轮次 1（sw 0.49/0.67、F̄ 0.0331）。明夜三选一：(a) redesign-3 新预注册（asym 变体，或论证放宽 PAIR_TOL——后者=改 gate，须显式拍板防 p-hacking）；(b) 组合拍板降规模（asym + switch_max 0.22 作限制报告 → 放行 Phase 3）；(c) 跳过 gate 直跑 experiment-long（前置未满足，不推荐）。publish 已完成：origin/main 推至 afa4abd（07:49 通道恢复后成功），本单元 commits 随后推送。
- **拍板结果（2026-10-23 08:15，本单元落档 DECISIONS）**：取 wrap 恢复点选项 (b)——**Phase 3（experiment-long T=50）规则 = asym（k_in=2/k_out=3）**，F2 以限制报告携带（gate 形式上未过：seed4 配对 +0.0110>0.005；switch_max 0.22 与 F̄≤0.0331 已过）。偏离预注册降规模文本所举 k=2 的理由：同批实测 asym 在每一根轴严格优于 k=2（switch max 0.22 vs 0.44、F̄ 0.0304 vs 0.0333、配对 5/5 种子不差于、forced 0 vs 1），asym 为预注册候选④a 且参数跑前冻结——选它不是事后调参。
- Phase 2 要点：F1 存活（max|ρ|=0.60，去趋势口径一臂 −0.88 贴线）；次判据①图像上不成立（合成 −41% 不外推，1 胜 1 负 1 平，spectral 均 F 反超 baseline）；主判据未达 F3 延续（triplet≡beta 动作 3/3 全同）；F2/F5 触发（F5 稳态 ~11× 训练耗时）

## 产物
- `workflow.json`——**生产管线登记（2026-10-23 design 单元）**：H1/预测/F1-F5（只引用
  pilot-design.md §7 预注册，不重定义）/最小可判别实验（F2 修复 gate + Phase 3 T=50 终判）/
  完成文件/各阶段回执（Phase 1、2 真实结果 backfill，source_mode=live）
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
- `results/pilot-f2fix-metrics.json`——**F2 修复正式实验（40 runs：6 唯一规则单元 × 5 种子 × 5 臂 + 1 别名单元）**
- `results/pilot-f2fix-seed{0..4}-{none,detrend}-tau{2.0,3.0}[-p1,-p3].log` + `results/pilot-f2fix-seed{0..4}-persist-tau2.0.log`——每单元完整日志（40 个）
- `results/pilot-f2fix-summary.md` + `results/pilot-f2fix-summary.json`——**F2 修复主表与验收判读（本阶段结论）**
- `experiments/analyze_f2fix.py`——F2 修复聚合分析（重写版：规则标签含 persist 轴 / F1 跨种子汇总预注册口径 / switch 双口径判读）
- `experiments/gating.py`、`experiments/train.py`——F2 修复候选实现（TrendAdjustedZ 去趋势 z / decide persist=k 持续期平滑 / --f2-fix --persist --tau-hi）
- `results/pilot-f2fix2-metrics.json`——**F2 修复第二轮正式实验（40 runs：8 唯一规则单元 × 5 种子 × 5 臂；config 含全部行为轴 f2_fix/persist/tau/band/k_in/k_out/ewma_alpha）**
- `results/pilot-f2fix2-seed{0..4}-{ctrl,p2,p3,band,asym,robust,ewma,dtb}.log`——每单元完整日志（40 个）
- `results/pilot-f2fix2-summary.md` + `results/pilot-f2fix2-summary.json`——**第二轮主表与验收判读（7/7 FAIL，asym 最近线）**
- `results/phase3-metrics.json`——**Phase 3 终判正式实验（25 runs：T=50 × 5 臂 × 5 种子，规则=asym i2o3）**
- `results/phase3-seed{0..4}.log`——每种子完整日志（5 个）
- `results/phase3-summary.md` + `results/phase3-summary.json`——**Phase 3 主表与终判判读（H1 falsified/F3 终判；F1/F2/F5 并列）**
- `results/phase3-cost-curve.png`——**F5 成本曲线（J+SVD+CKA vs 训练耗时，T=10/25/50）**
- `results/phase3-costprobe-T25-metrics.json` + `results/phase3-costprobe-T25-seed0.log`——F5 成本曲线第三点探针（T=25 seed0，非判据网格）
- `results/_phase3-driver.log`——Phase 3 网格驱动运行记录
- `experiments/run_phase3.py`——Phase 3 网格驱动（预注册单元表 + 可恢复 skip-done，签名含 T）
- `experiments/analyze_phase3.py`——Phase 3 聚合分析（同口径：主判据配对 bootstrap/次判据② T 对比/F1 跨种子汇总/F2 限制报告/F5 成本曲线/F3 结构+槽位饱和）
- `results/_f2fix2-driver.log`——网格驱动运行记录（断点/续跑/skip 证据）
- `experiments/run_f2fix2.py`——第二轮网格驱动（预注册单元表 + 可恢复 skip-done）
- `experiments/analyze_f2fix2.py`——第二轮聚合分析（同口径：标签含全部行为轴 / F1 跨种子汇总 / switch 双口径 / 预注册预测对照）
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

## F2 修复读数（experiment-redesign，正式结论域，5 种子 × 5 臂；验收 gate=预注册）

对照 none/p1/tau2（原规则同批重跑）spectral/triplet：F̄=0.0331（=归档基准），switch 0.49/0.67
（复现 Phase 1 触发）。验收=switch<0.3（主口径 per-seed max；次口径 mean）+ F̄≤0.0331 + 配对不劣化。

| 规则（候选） | spectral F̄ | spectral sw mean/max | triplet F̄ | triplet sw mean/max | 总判(主/次) |
|---|---|---|---|---|---|
| none/p1/tau2（对照） | 0.0331 | 0.49/0.67 | 0.0331 | 0.49/0.67 | （触发态） |
| ③ persist k=2 | 0.0333 | 0.27/0.44 | 0.0333 | 0.27/0.44 | FAIL/FAIL |
| ③ persist k=3 | 0.0433 | 0.13/0.22 | 0.0433 | 0.13/0.22 | FAIL/FAIL |
| ① detrend 单独 | 0.0451 | 0.29/0.56 | 0.0257 | 0.36/0.44 | FAIL/FAIL |
| ①+③ detrend k=2 | 0.0500 | 0.13/0.22 | 0.0311 | 0.22/0.22 | FAIL/FAIL |
| ② tau=3（敏感性） | 0.0454 | 0.27/0.67 | 0.0454 | 0.27/0.67 | FAIL/FAIL |
| ②+③ tau=3 k=2 | 0.0523 | 0.09/0.22 | 0.0466 | 0.13/0.22 | FAIL/FAIL |

- **6/6 候选未过预注册 gate（双口径都 FAIL）**。按 workflow verdict_mapping → redesign。
  k=2 距 PASS 最近（F 无损、switch mean 腰斩）但 seed2 max 0.44；k=3 稳但丢 F（seed1/4 漏全部告警）。
- **persist trade-off（k=1/2/3）**：switch mean/max 0.49/0.67→0.27/0.44→0.13/0.22；F̄ 0.0331→0.0333→0.0433；
  isolate 14→11→9——持续性平滑用灵敏度换稳定性，单调无免费午餐。
- **detrend 机制（新事实）**：去趋势后三通道全部开始越界（drift 0/40→7-11/40；beta 0-1/40→11-14/40）
  → cka/beta 臂从从不告警变频繁告警（beta switch 0.33-0.67，F2 新违例）、triplet 槽位耗尽
  （detrend 单独 forced 8 次）；spectral F 显著变差（+0.0169，CI95 不含 0）。detrend 换的是告警结构，不是修 F2。
- **F1 未触发（全规则）**：跨种子汇总 max|ρ|=0.48-0.72（旧草稿 0.9667 系单种子 n=9 假高，已按预注册口径重算）。
- F5 不变：cost_ratio ~0.34-0.39（规则层修复不改变触发计算成本）。

## F2 修复第二轮读数（experiment-redesign-2，正式结论域，5 种子 × 5 臂；验收 gate=预注册不变）

对照 none/p1/tau2/b0.3 同批重跑逐位复现轮次 1（spectral switch 0.49/0.67、F̄ 0.0331）。
验收=switch<0.3（主口径 per-seed max；次口径 mean）+ F̄≤0.0331 + 同批配对每种子 ΔF≤0.005。

| 规则（候选） | spectral F̄ | spectral sw mean/max | triplet F̄ | triplet sw mean/max | 总判(主/次) |
|---|---|---|---|---|---|
| none/p1/tau2/b0.3（对照） | 0.0331 | 0.49/0.67 | 0.0331 | 0.49/0.67 | （触发态） |
| ④a asym i2o3 | **0.0304** | **0.22/0.22** | **0.0304** | **0.22/0.22** | FAIL/FAIL（仅 seed4 配对 +0.0110>0.005） |
| ④a' band0.5 | 0.0359 | 0.47/0.67 | 0.0359 | 0.47/0.67 | FAIL/FAIL |
| ④b robust z | 0.0444 | 0.24/0.44 | 0.0427 | 0.29/0.44 | FAIL/FAIL（beta 臂 sw 0.44 新违例） |
| ④b' EWMA z | 0.0392 | 0.33/0.56 | 0.0385 | 0.33/0.56 | FAIL/FAIL |
| ④c detrend-bound | 0.0451 | 0.29/0.56 | 0.0460 | 0.29/0.56 | FAIL/FAIL |
| 家族 none/p2（k=2 重跑） | 0.0333 | 0.27/0.44 | 0.0333 | 0.27/0.44 | FAIL/FAIL（=轮次 1） |
| 家族 none/p3（k=3 重跑） | 0.0433 | 0.13/0.22 | 0.0433 | 0.13/0.22 | FAIL/FAIL（=轮次 1） |

- **7/7 候选未过预注册 gate（双口径都 FAIL）**；asym 最近线：switch 0.22/0.22 全种子一致
  （唯一双口径过线）+ F̄ 0.0304≤0.0331 + isolate 15/forced 0（对照 14/2），仅 seed4 配对
  ΔF=+0.0110>0.005（mean Δ=-0.0027，CI95[-0.0086,+0.0048] 含 0）。
- **新事实：非对称持续期打破 persist 单调 trade-off**。k=1/2/3：switch 0.49/0.67→0.27/0.44
  →0.13/0.22 vs F̄ 0.0331→0.0333→0.0433（单调交换）；asym 2/3 同时改善两侧——进 isolate 需
  2 连续告警（保灵敏度）、退需 3 连续不告警（粘性退出）→ 单尖峰不再制造 i,m 振荡，且隔离
  块后移/延长仍保护任务。seed4 代价机制：进入延迟错过 t5 单尖峰 → 隔离块 t6-t8（对照
  t5/t8/t9），早期任务遗忘抬高（t1 0.074→0.116）。
- **z 族修法（robust/EWMA）一律以 F 劣化为代价**：robust 钝化早期尖峰（bound 越界 13→8/40，
  F̄ 0.0444，配对 Δ+0.0113 CI95[+0.0020,+0.0242] 显著变差）且激活 beta 通道（越界 1→3-4/40，
  beta 臂 switch 0.44 新违例）；EWMA 告警局部化（bound 13→8/40）但稳态 z≈2.4 贴线 → seed3
  switch 0.56。**detrend-bound 不直接激活 drift（0/40）但 F 仍劣化**（0.0451）——劣化来自
  动作落点/路径效应而非通道激活，坐实「换告警结构 ≠ 修 F2」。
- band 加宽 0.3→0.5 无效（预测应验）：0.47/0.67、F̄ 0.0359——hysteresis band 内 z 多 <1.5。
- F1 未触发（全规则跨种子汇总 max|ρ|=0.6659-0.6818<0.9）；F5 不变（cost_ratio 0.37-0.41）。
- **拍板**：Phase 3 规则=asym + F2 限制报告（详见 DECISIONS 2026-10-23 判读条）。

## Phase 3 终判读数（experiment-long，正式结论域，5 种子 × 5 臂；规则=asym i2o3，T=50）

| 臂 | F̄±sd | F̄/种子 | acc | switch(mean/max) | isolate(forced) | cost_ratio |
|---|---|---|---|---|---|---|
| baseline | 0.3302±0.0183 | 0.3370/0.3508/0.3337/0.3009/0.3288 | 0.6596 | 0/0 | 0(0) | 3.144 |
| cka | 0.3302±0.0183 | 同上（从不告警） | 0.6596 | 0/0 | 0(0) | 3.153 |
| spectral | **0.2928**±0.0232 | 0.2940/0.3276/0.2966/0.2648/0.2812 | 0.6980 | 0.04/0.04 | 15(31) | 3.150 |
| beta | 0.3302±0.0183 | 同上（从不告警） | 0.6596 | 0/0 | 0(0) | 3.157 |
| triplet | **0.2928**±0.0232 | 同 spectral | 0.6980 | 0.04/0.04 | 15(32) | 3.096 |

- **H1 主判据：FAIL → verdict=falsified（F3 终判，负结果）**。triplet vs spectral
  ΔF̄=0.0000（动作序列 5/5 种子完全一致）；vs cka/beta ΔF̄=+0.0374（配对 CI95
  [+0.0299,+0.0439] 不含 0、5/5 种子同向）但 **<5pp 条款**；vs baseline 同幅（次判据①
  合成仍成立但幅度不足）。预注册口径（Δ=F(单量)−F(triplet)，CI95 不含 0 且 ≥5pp）。
- **结构解释（新事实）**：triplet 全部增益来自 bound 单通道——spectral/triplet 动作
  逐位一致；drift/beta 通道 240 决策点仅告警 1/4 次（bound 54 次）。**槽位饱和**：
  3 个 isolate 槽在 task 4-8 耗尽（slots_used=4/4），中晚期告警全落 forced merge
  （每种子 1-10 次）——机制容量假设稀疏告警，长序列运行在饱和区。
- **次判据②**：ΔF̄ T=10→T=50：vs cka 0.0226→0.0374、vs beta 0.0198→0.0374、
  vs spectral 0→0（同规则 asym T=10：0.0253→0.0374）。对从不告警臂增益随 T 放大
  （方向合预测），但 5pp 从未达到、vs spectral 恒 0——三元组无决策增量与 T 无关。
- **F1 未触发**：跨种子汇总（n=245）max|ρ|=0.7331（drift~bound）<0.9——三量不退化，
  H 因「无增量」而非「三量退化」被驳倒。
- **F2：T=50 未触发**（switch 0.04/0.04）；**F2 限制报告（并列）**：T=10 同批 gate
  形式上未过（asym seed4 配对 ΔF=+0.0110>0.005；switch_max 0.22 与 F̄ 0.0304 已过），
  Phase 3 规则=asym 系组合拍板降规模（候选④a 跑前冻结），非事后调参。
- **F5 严重触发**：cost_ratio 0.38（T=10）→1.19（T=25）→3.10（T=50）。成本曲线
  （results/phase3-cost-curve.png）：触发 J+SVD+CKA 0.55→4.04→21.5s/run（实测 7.4×/39×
  vs 结构 T(T−1)/2 预期 6.7×/27.2×）；训练 1.43→3.38→6.95s（线性）。论文必须报告。
- 复证：seed0 同批重跑行为字段逐位 identical（GPU 确定性复证，同轮次 1/2 惯例）。

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
    # F2 修复（experiment-redesign）：规则单元 = (--f2-fix {none,detrend,persist},
    # --persist {1,2,3}, --tau-hi {2.0,3.0})；40 runs 汇总：
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase pilot --seed <S> \
      --f2-fix <fix> --persist <k> [--tau-hi <tau>] \
      --out results/pilot-f2fix-metrics.json --log results/pilot-f2fix-seed<S>-<fix>-tau<tau>[-p<k>].log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_f2fix.py \
      --in results/pilot-f2fix-metrics.json --md results/pilot-f2fix-summary.md \
      --json results/pilot-f2fix-summary.json   # F2 修复验收判读
    # F2 修复第二轮（experiment-redesign-2）：规则单元 = (--f2-fix {none,asym,robust,ewma,
    # detrend-bound}, --persist {1,2,3}, --tau-hi, --band, [--k-in/--k-out], [--ewma-alpha])；
    # 8 唯一单元 × 5 种子 = 40 runs（驱动可恢复，skip-done）：
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/run_f2fix2.py \
      --out results/pilot-f2fix2-metrics.json   # 或逐单元 train.py --phase pilot --seed <S> \
      --f2-fix <fix> --persist <k> --band <b> [--k-in 2 --k-out 3] [--ewma-alpha 0.3] \
      --out results/pilot-f2fix2-metrics.json --log results/pilot-f2fix2-seed<S>-<tag>.log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_f2fix2.py \
      --in results/pilot-f2fix2-metrics.json --md results/pilot-f2fix2-summary.md \
      --json results/pilot-f2fix2-summary.json   # 第二轮验收判读（同口径）
    # Phase 3 长序列终判（experiment-long）：规则=asym（--k-in 2 --k-out 3），T=50 经
    # train.py --T 覆盖（PHASES 无 T=50 入口）；25 runs 驱动可恢复：
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/run_phase3.py \
      --out results/phase3-metrics.json   # 或逐种子 train.py --phase pilot --seed <S> \
      --T 50 --f2-fix asym --persist 1 --tau-hi 2.0 --band 0.3 --k-in 2 --k-out 3 \
      --out results/phase3-metrics.json --log results/phase3-seed<S>.log
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/analyze_phase3.py \
      --in results/phase3-metrics.json --md results/phase3-summary.md \
      --json results/phase3-summary.json   # Phase 3 终判判读（含 F5 成本曲线）

## git
- 本地仓库 main 分支，今晚 8 commits（建档 785ea69 / smoke 1f2d427 / 收工 ed238b7 /
  校准 v2 8573eb5 / pilot 结果+文档 119d61a / STATUS SHA 修正 ad57ffc / Phase 2 结果+脚本
  5a5aeae / Phase 2 判读落档 24f42c2 / design 管线登记 1ea4ba2 /
  **experiment-redesign 完成 d767fc2**）；2026-10-23 07:39 wrap 单元已建公开仓并推送：https://github.com/3gediana/cert-gating-multilayer-perturbation（origin/main，12 commits）
- experiment-redesign-2：预注册+实现 afa4abd（07:32 台账冻结）→ wrap 并收 43ea954（首版
  结果+恢复点）→ **本单元判读+拍板落档 291d180**（DECISIONS 判读条/STATUS 第二轮读数/
  workflow 回执；重生结果覆盖首版——行为逐位 identical，仅补 config 行为轴）。
- **push 仍受阻（2026-10-23 09:3x 复试）**：`git push origin main` 再次挂起（100s 超时杀，
  exit 124；ls-remote 正常返回——通道对读通、对写挂起）。本地 main=**ca63090** 完好；
  ls-remote origin/main=**4133705**（wrap 单元推后一直停留；afa4abd 起各 commit 待推）。
  遗留独立单元：通道恢复后 push（publish=github-push 已授权）。
- **push 再挂（2026-10-24 00:1x→02:2x 轨迹）**：00:1x 复验通道恢复（ls-remote 通 +
  push --dry-run exit 0）→ 02:2x 组合统筹第 21 轮真实 push 四路全败：①默认配置 100s
  超时（credential.helper=manager 疑似 GUI 交互挂起）；②GIT_TERMINAL_PROMPT=0 +
  GCM_INTERACTIVE=Never → `TLS connect error: unexpected eof while reading`（写通道
  被断，读通道 ls-remote 仍通）；③显式走 http.proxy=127.0.0.1:7897（仓库级已配置）
  → 同样 TLS unexpected eof；④SSH 探测 `ssh -T git@github.com` → Permission denied
  (publickey)，~/.ssh/config 无 github 条目，id_ed25519 未注册到 3gediana。
  判定：HTTPS 写通道间歇性被干扰（与 09:3x 同模式），SSH 备用路径未配置——
  本地 main=**a74fcf7** 完好（8 commits 待推），通道恢复后 push（publish 已授权）；
  备选根治：白天把 3gediana 的 SSH key 注册到 GitHub 或换 gh CLI credential。
- **push 第 5 路实证（2026-10-24 02:4x，组合统筹第 27 轮）**：gh credential helper
  （gh 已登录 3gediana，repo scope，keyring）替代 GCM 后 push 仍 TLS unexpected eof——
  **凭据层排除，写通道（receive-pack POST）传输层干扰坐实**（读通道 ls-remote 同刻
  返回 4133705 正常、curl 双路 200；本轮开工时 ls-remote 一度 fatal 数分钟后自愈，
  通道 flap）。另 PROJECT.md 项目档案首次入库（437576d），本地 main=**437576d**，
  待推 10 commits。夜间不再重试；白天通道恢复后
  `git -c credential.helper='!gh auth git-credential' push origin main` 直推，或注册 SSH key。
- **PUSH 成功（2026-10-24 03:5x，组合统筹第 38 轮）**：代理 7897 路探测发现通道恢复——
  读路 ls-remote 通（4133705）+ `push --dry-run` 全过（4133705..6610496 negotiate 完成）→
  真实 push `GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=Never git -c http(s).proxy=http://127.0.0.1:7897 push origin main`
  **exit 0，4133705..6610496，11 commits 全部落地**；ls-remote 复核 origin/main=**6610496**
  =本地 HEAD，待推清零。同刻直连读路仍 TLS eof（FLAP 实锤：故障在直连路 TLS 间歇断，
  代理路间歇恢复；第 21 轮"显式代理也败"是当刻代理路同陷故障，非代理路固有封锁）。
  GCM_INTERACTIVE=Never 亦排除 GCM GUI 挂起因素。**遗留 push 单元完结**；后续同步
  一律走代理路 + GCM_INTERACTIVE=Never（仓库级 http(s).proxy=127.0.0.1:7897 本已配置）。

## 下一步（workflow 管线）
- **experiment-long（Phase 3 终判）已完成（2026-10-23 09:2x）：H1 合成尺度终判 falsified**
  （F3，负结果）——25 runs（T=50 × 5 臂 × 5 种子，规则=asym），主判据 FAIL（vs spectral
  ΔF̄=0；vs cka/beta +0.0374 CI95 不含 0 但 <5pp）；F1 未触发（0.7331）；F2 T=50 未触发
  （0.04，T=10 gate 未过状态限制报告并列）；F5 严重触发（cost_ratio 3.10，成本曲线
  results/phase3-cost-curve.png）。判读落档 DECISIONS/STATUS/workflow.json。
- 论文负结果成文（建议主路径）：三平台一致结论——三元组门控无单量外增量（合成 T=10/T=50
  动作与 spectral 逐位一致；Split-CIFAR10 与 beta 全同），门控本身有效但对分布敏感
  （合成 −41%/+0.0374 成立、图像 1 胜 1 负 1 平）；F2/F5 为独立报告项（成本曲线必附）。
- redesign-3（备选，未启用）：若需继续修 F2 gate（asym 变体如 k_out=4，或显式拍板放宽
  PAIR_TOL 的防 p-hacking 论证），须新预注册另开单元；对 H1 终判无影响（H1 已驳倒）。
- ~~遗留独立单元：origin 推送本课题 commits~~ **已完结（2026-10-24 03:5x 第 38 轮）**：
  代理路 push 成功，origin/main=6610496（43ea954/291d180/b2aeea4/36be619/ca63090/a74fcf7/
  437576d/6610496 全部在远端）。

- experiment-long（Phase 3 终判）：预冻结+管线步骤 36be619（train.py --T 覆盖 smoke 验证 +
  run_phase3.py + DECISIONS 协议冻结条）→ **本单元判读落档**（25 runs 804s + T=25 成本探针 +
  seed0 重跑逐位复证 + analyze_phase3 判读：H1 falsified/F3 终判；DECISIONS 判读条 / STATUS
  Phase 3 读数 / workflow 回执）。

## 阻塞
- 无硬阻塞。**H1 已在合成尺度终判 falsified（F3）**：三元组门控相对单量（bound/spectral）
  无决策增量（T=10 与 T=50 均为 ΔF̄=0，动作逐位一致），对 cka/beta/baseline 的增益全部来自
  bound 单通道且 <5pp；F1 未触发（三量不退化，H 因无增量被驳倒）。次判据①（门控有效）
  在合成上成立（+0.0374，CI95[+0.0299,+0.0439]，5/5 种子同向）但图像分布上不成立
  （Phase 2）。F2 未清状态（T=10 seed4 配对 +0.0110>0.005）随全部结论并列。剩余可选方向：
  论文负结果成文（主）；redesign-3（显式放宽 PAIR_TOL 的防 p-hacking 论证或 asym 变体）
  已非判据需要；origin 推送遗留（通道恢复后补推 43ea954/291d180/b2aeea4 起各 commit）。
  撞道提示：STATUS.md/workflow.json 为并行会话热点文件，落笔前须重读+git diff 核对。
