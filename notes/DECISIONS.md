# 决策台账 — 证书门控-多层扰动度量

> 拍板都记；方案级否决记（带理由，防止将来被自然再次提出时重蹈）；操作级纠正（别多做事/别抢跑类）不进台账——纠正改完行为就消失，不复读。

## [taste] ICLR 2025 拒稿死法（taste_mine 抽样）：「两个成熟问题的组合」若无必要性论证（为什么这个组合有真实场景、组合后产生什么任一单领域没有的新洞见），评审直接判无价值——死法排第一。延续到我们的多层证书门控：组合 CKA/谱界/泄漏项三层量，贡献点不能是「三层比一层好」，必须是完备性论证（哪一层漏了会导致什么失效模式，每层对应一种独立的失效通道）否则同样死。附带两个同类雷：①声称不存历史数据但门控量依赖任务边界锚/梯度记忆=间接数据访问（claim 与实现一致性）；②谱量计算（SVD/power iteration）成本必须实测报告，评审会把可扩展性当独立缺陷打。
- 关联：证书门控-多层扰动度量
- 理由：多层证书门控正是「谱族量+CKA+门控」的组合型贡献，2026-10-21 抽样拒稿（DGP 对抗鲁棒×持续学习）死因与该结构同型，立项设计时必须先过完备性论证与 claim 一致性自检

## [2026-10-22] 门控语义定为 rollback-retrain（触发器因果性）
- 方案：任务 i 默认并入 trunk0 试训练 → 任务边界计算三量 → 告警则回滚 trunk（参数快照）→ 隔离槽重训任务 i；槽位耗尽强制并入。
- 理由：触发量必须决策时点可计算且因果——不能用训练后结果（如遗忘）当触发器，否则循环论证；试训练+回滚让"若并入会怎样"成为可测的因果量。
- 动作空间 pilot 只行使 {merge, isolate}；扩展容量/重算全局结构在规则中留接口不触发（有界单元取舍，pilot-design.md §8 范围声明）。

## [2026-10-22] 三量操作化定义（pilot-design.md §3）
- S_i = linCKA(trunk0 对 A_{i-1} 的前后特征)，漂移 d=1−S；B = ||ΔW||₂·max_r ||J_r||_*（vmap 逐样本梯度矩阵核范数）；β = max_r 锚梯度能量落 span{ΔW_{i-1},ΔW_i} 的 Frobenius 比例；ΔW 取试训练量。
- 理由：三通道必须可分离（表征几何 / 最坏情况幅值 / 方向对齐），否则 F1 退化检查（两两 |ρ|>0.9）无意义。

## [2026-10-22] 数据访问声明（claim 一致性，防 taste 雷①）
- 门控三量依赖每任务锚集（smoke 32 / pilot 64 条训练样本）+ 锚梯度矩阵缓存 = 显式记忆访问；论文 claim 不得称 rehearsal-free，所有臂同记忆预算。

## [2026-10-22] 失败阈值预注册（pilot-design.md §7）
- F1 两两 |ρ|>0.9 → 驳倒；F2 action-switch rate >30%；F3 等参数量无显著增益；F5 触发计算 > 训练耗时 20%。
- smoke 实测 F5 贴线（~0.20）——phase 1 提高 steps 后预计降至 ~0.12，列为盯防项而非失败。

## [2026-10-22] 合成难度校准 + GPU 一次性开销教训
- 难度：k=4/noise=0.5 首跑 acc 饱和 1.0、遗忘恒 0（对比区不存在）→ 定 k=10/noise=1.0/drift=0.5，baseline F=0.0375 进入非零遗忘区。
- 工程教训：linear_cka 首调 ~0.12s 的一次性开销污染 F5 臂间比较（首臂假高 2.4×）→ 预热必须覆盖全部触发函数，已在 train.py 预热块。

## [2026-10-22] 难度校准 v2：pilot width 256 否决，定 width 64
- 方案：Phase 1 pilot 配置由 T=10 × width 256 改为 T=10 × width 64（steps 100，drift 0.5，noise 1.0，k 10，anchor 64——与 smoke 同难度协议仅放量）；train.py 增加 --width/--drift/--noise/--k/--steps 校准覆盖参数。
- 理由：width 256 时 baseline 遗忘被容量压到 0.003~0.005（近饱和、spectral 臂从不告警、门控路径空转），对比区不存在——pilot-design.md §4 的"baseline 必须有非零遗忘"是有效度前提，难度校准优先于宽度规格。校准扫描（seed 0）：w256/d1.0 F=0.0030；w128/d1.0 F=0.0140；w64/d1.0 F=0.0342；w64/d0.5 F=0.0534（acc 0.9436）。width 256 同时 cost_ratio 1.5+（F5 双杀）。证据 results/calibration-sweep.txt。

## [2026-10-22] Phase 1 pilot 判读（5 种子，width 64）
- F1 未触发：跨种子汇总（n=45/臂）max|ρ|=0.68，去趋势后 |ρ| 全 <0.35——三量不退化，H 存活。注意 width 256 单种子曾现 ρ(drift,bound)=−0.9，n=9 + 趋势假象，n=45 回落——F1 判读必须跨种子汇总，单种子不可判（smoke 的预判成立）。
- 主判据未达（F3 路径）：triplet 与 spectral 动作序列 5/5 种子完全一致（ΔF̄=0）；vs cka/beta Δ≈0.02 显著（CI 不含 0、5/5 同向）但 <5pp 条款。校准难度下只有 bound 通道告警（cka 0/50 决策从未告警，beta 仅 seed2 一次）→ 三元组相对单量在 T=10 难度无决策增量。按 §7 F3：pilot 尺度 H 未获支持，升级长序列 Phase 3 再判（T=50），仍无则驳倒。
- 有效事实（可写进论文的负面骨架）：spectral 单臂门控 F 0.0557→0.0331（相对 −41%，5/5 种子，配对 bootstrap CI [0.0158,0.0297]）——门控本身有效（次判据①），三元组未增量。
- F2 触发（spectral/triplet switch 0.49/0.67 >0.3）：根因假设 bound 随任务序号单调增长 + RunningZ 全历史均值滞后 → 边界穿越振荡；hysteresis band 0.3 太窄。按预注册"重设计后再判"——下一单元先修（去趋势 z / 提 τ / 持续期平滑三候选），验收 switch<0.3 且 F 不劣化。
- F5 触发（cost ~0.35 >0.2，SVD 主导）：触发成本随 T 超线性（J 覆盖全部锚任务 ~O(T²m²P) vs 训练 O(T·steps)）；width 256 时 1.5+。必须作为可扩展性缺陷报告（taste 雷②）。
- 工程坑记录：校准覆盖参数初版 width 按 float 传入 nn.Linear 报 TypeError（width/k/steps 必须 int）——已修。

## [2026-10-22] Phase 2 判读（Split-CIFAR10 3 种子，width 256）
- F1 存活：max|ρ| 原始 0.6014（bound~beta，beta/triplet 臂）；去趋势最大 −0.8811（drift~beta，n=12 小样本噪声大，未越 0.9 线）——三量在图像上仍可分离，H 未驳倒。但余量比 pilot（0.68 / <0.35）变薄。
- **次判据①不跨分布**：triplet vs baseline ΔF̄=+0.0129（CI95 [−0.0234,+0.0622] 含 0；1 胜 1 负 1 平）；spectral 均 F 0.1190 > baseline 0.1121（门控反而略差）；seed1 beta/triplet −40%（0.1550→0.0928）vs seed2 +34%（0.0698→0.0932）符号翻转。合成上 5/5 一致 −41%（CI 不含 0）的显著性在图像分布上不保持——**门控有效性对数据分布敏感**，合成基准显著性不可外推到图像。
- 主判据未达（F3 路径延续）：triplet 与 beta 动作序列 3/3 种子全同（ΔF=0）；vs spectral Δ=0.0199（CI 含 0）；vs cka/baseline Δ≈0.013（CI 含 0）。
- 触发结构（新事实）：drift 通道 0/27 决策点告警（z 恒 ≤0——MLP trunk 跨任务表征随序列训练变稳，与合成旋转冲突结构相反）；6 次告警全部来自 task3/4 的 bound 或 beta 尖峰（z 2.5-7.9）；隔离收益取决于隔离槽训练充分度（100 步×2000 样本在图像上欠拟合 → seed2 回滚+重训反噬 +34%）。
- F5 严重：稳态 cost_ratio ≈ 11×（P=3.4M 是合成 31K 的 110×）；baseline 臂首调 ~22×（首次 cuSOLVER/分配器开销，3 种子一致，根因未定位）。论文必须报告（taste 雷②）。
- 工程记录：CIFAR 官方源 ~60KB/s（301→cave.cs.toronto.edu）；HF 镜像 datasets/fffan/Temp ~12MB/s 但为 ZIP（内含 Cifar10/cifar-10-batches-py）——预置解压后 torchvision 完整性校验通过（校验对象是解压文件 md5 而非 tar，CIFAR10(download=False) 可用）。

## [2026-10-23] workflow 管线登记（candidate-design 单元）：准入成立，判据权归预注册
- 驱动器 06:17 派发 candidate-design（快权重课题 stop 后队列空，落到本课题）：workflow.json 登记 H1/预测/失败条件/最小可判别实验/完成文件，STATUS 标 active/design。准入显然成立（Phase 0-2 真实结果在盘、F3 预注册路径未走完、组合 2/3 项目）→ next_action=continue。
- 关键拍板：workflow.json 对 F1-F5 只**引用** pilot-design.md §7 预注册、不重定义阈值——PORTFOLIO 04:59 记录的「泛化 workflow 抢占预注册协议」担忧以"引用而非重写"落进 schema。管线 ready 阶段=experiment-redesign（F2 修复三候选，验收 switch<0.3 且 F̄ 不劣化 0.0331），后接 experiment-long（Phase 3 T=50 终判）。
- 图像触发惰性的修法（τ 分布校准/尖峰持续期确认/隔离槽步数放大）随 F2 修复同轮验证，但 τ 分布校准只能作敏感性分析报告——不得事后调参改预注册主判据。

## [2026-10-23] F2 修复判读（experiment-redesign 阶段）：预注册三候选 6/6 全部未过验收 gate
- 网格：6 个唯一规则单元 × 5 种子 × 5 臂（pilot T=10 width64 校准版，40 runs）+ 1 别名单元。
  对照 = none/p1/tau2（原规则同批重跑：spectral switch 0.49/0.67、F̄=0.0331——逐位复现
  Phase 1 触发读数与归档基准，对照有效）。别名 persist/p2/tau2≡none/p2/tau2（f2_fix=persist
  只是 --persist 的别名，行为字段逐位相同）——作 GPU 确定性复证，不进分析。
- 验收（workflow.json gate_step 预注册）：switch<0.3（主口径=per-seed rate 跨种子 max，
  与 analyze_pilot.py 的 F2 实现一致；次口径=mean，双口径都判）+ F̄≤0.0331 + 同批配对
  每种子不劣化（>0.005 视劣化）。
- **总判：6/6 候选 FAIL（主/次双口径都 FAIL）**。明细（spectral/triplet 臂）：
  - ③ persist k=2：F̄ 0.0333（配对 Δ+0.0002，CI95[-0.0071,+0.0096] 含 0——F 无损），
    switch 0.27/0.44——seed2 仍 0.44>0.3（主口径败）；seed4 配对 +0.0168（漏 1 次隔离）。
    距 PASS 最近的一格。
  - ③ persist k=3：switch 0.13/0.22 ✓ 但 F̄ 0.0433 ✗（配对 Δ+0.0102；seed1/4 告警簇需
    连续 3 边界确认 → 完全不隔离，F 回落到 baseline 0.0960/0.0486）。
  - ① detrend 单独：spectral sw 0.29/0.56 ✗、F 0.0451 ✗；beta 臂 sw 0.67 ✗（新违例）。
    ①+③：spectral sw 0.13/0.22 ✓ 但 F 0.0500 ✗（配对 Δ+0.0169，CI95[+0.0067,+0.0276]
    不含 0——显著变差）；triplet sw 0.22/0.22 ✓ F 0.0311 ✓，仅 seed4 配对 +0.0074 越容忍
    且 beta 臂 0.33 ✗。
  - ② tau=3（仅敏感性分析）：单独 sw 0.27/0.67 ✗ F 0.0454 ✗；+③ sw ✓ F 0.0523 ✗。
    阈值类修法一律以 F 劣化为代价——坐实"只能作敏感性分析"的约定。
- persist 家族 trade-off（k=1/2/3）：switch mean/max 0.49/0.67→0.27/0.44→0.13/0.22；
  F̄ 0.0331→0.0333→0.0433；isolate 14→11→9。单调：持续性平滑用灵敏度换稳定性。
- detrend 机制（新事实）：去趋势后三通道全部开始越界（drift 0/40→7-11/40，beta 0-1/40
  →11-14/40）——drift 随任务序号单调下降、原始 z 恒负，去趋势后晚期残差转正 → cka/beta
  臂从"从不告警"变频繁告警（beta switch 0.33-0.67，F2 新违例），triplet 槽位耗尽
  （detrend/p1 forced 8 次）。detrend 不是"修 F2"而是换了一组告警结构。
- F1：全部规则跨种子汇总 max|ρ|=0.48-0.72 <0.9 → 未触发。此前草稿里的 0.9667 是单种子
  n=9 假高（§8 判读细则预判的现象再次应验）——已按预注册口径（跨种子汇总）重算改写。
- 处置：按 workflow verdict_mapping"F2 修复失败 → redesign"。下一单元预注册候选④集
  （先写台账再跑，防事后调参）：a) 非对称持续期（进 isolate k=2/退 k=3）或加宽
  hysteresis band；b) 稳健 z（median/MAD）或 EWMA z（隐式去趋势、不激活全部通道）；
  c) 混合：仅 bound 通道去趋势。验收 gate 不变。降规模选项（待组合拍板）：Phase 3 直接用
  k=2 规则、switch_max 0.44 作限制报告。
- 工程记录：f2_fix=persist 与 none+--persist 行为等价（别名，旧版分析标签碰撞会静默
  覆盖单元——标签必须含 persist 轴）；write_file 拒覆盖已读文件时经 scratch+cp 落盘。

## [2026-10-23] F2 修复第二轮预注册（experiment-redesign-2）：候选④集 + 验收 gate（跑前冻结）
- 背景：轮次 1（commit d767fc2）三候选 6/6 未过 gate。从 round-1 归档 records 读出的
  机制（候选设计依据，非本轮新数据）：①告警是单边界尖峰——z 跳 >τ 一个任务后回落到
  ~1-1.5 → 动作序列 i,m,i,m 振荡；②连续 isolate 边界各占一个槽位（T=10 仅 3 槽）→
  粘性规则会加速槽位耗尽、forced merge 反制造 switch（seed2/persist k=2 的 max 0.44
  部分源于此——t9 是 forced merge 而非规则切换）；③detrend 失败=直接统计效应（drift
  原始 z 恒负、去趋势后晚期残差转正→越界）+ 动作路径效应。
- 候选④集（参数全部跑前冻结；本单元内任何调参都违反预注册——要改只能作为新一轮
  预注册另开单元）：
  1. `asym`（④a 非对称持续期）：进 isolate 需连续 k_in=2 告警、退需连续 k_out=3 不告警
     （RunningZ / tau2 / band0.3 / persist=1）。假设：振荡主因"进易退易"——放慢退出、
     进入保持 2（保 k=2 的 F 灵敏度）。预测：switch 低于 persist k=2（0.27/0.44）、
     F̄≈k=2 水平（~0.033）；风险：粘性 isolate 加速槽位耗尽 → forced merge。
  2. `none/p1/b0.5`（④a' 加宽 hysteresis band 0.3→0.5）。假设：band 内维持 isolate
     可吸收尖峰后回落。预测：无效或边际（round-1 多数 override 点 z<1.5）；风险：
     override 与槽位耗尽碰撞（seed2 控制组 t8 已见 m*h）。
  3. `robust`（④b 稳健 z=median/MAD；min 3 样本；MAD=0 时用相对中位数地板防除零）。
     假设：中位数/MAD 对 ramp 末端不敏感 → 晚期持续越界消除。预测：告警减少、F̄ 向
     baseline 回落（早期尖峰也被钝化——seed2 t3 z 3.74→~0.9 量级）；风险：丢 F。
  4. `ewma`（④b' EWMA z；alpha=0.3；偏差修正方差；min 3 样本）。假设：局部自适应均值
     =隐式去趋势——只对局部水平偏离告警、不做全局去趋势 → 不直接激活 drift/beta
     （针对 detrend 全通道激活的失败机制）。预测：告警局部化、晚期不再持续越界；
     风险：纯 ramp 上稳态 z≈2.4 贴 τ=2 → 可能出现新的贴线振荡。
  5. `detrend-bound`（④c 混合）：仅 bound 通道 TrendAdjustedZ，drift/beta 保持 RunningZ。
     假设：只修有趋势问题的 bound 通道、drift 保持"原始 z 恒负不告警" → 不重演 detrend
     的全通道激活。预测：bound 振荡修复且 drift/beta 不直接越界；风险：动作路径效应
     仍改变 drift/beta raws（round-1 detrend 见 drift 7-11/40、beta 11-14/40 越界）。
- 同批对照 + 家族上下文（同批配对的前提）：`none/p1/tau2/b0.3`（对照，兼与 round-1
  归档值逐位复证 GPU 确定性）；`none/p2`、`none/p3`（round-1 两格同批重跑，把 asym
  放上 persist trade-off 曲线 k=1/2/3 + 2/3 非对称）。
- 网格：8 个唯一规则单元 × 5 种子 × 5 臂 = 40 runs（pilot T=10 width64 校准版，
  commit d767fc2 同协议）。产物：results/pilot-f2fix2-metrics.json +
  results/pilot-f2fix2-seed{0..4}-{ctrl,p2,p3,band,asym,robust,ewma,dtb}.log +
  experiments/analyze_f2fix2.py 汇总（同口径：标签含全部行为轴/F1 跨种子汇总/
  switch 双口径）。
- 验收 gate（不变=workflow.json gate_step + pilot-design.md §7 F2 预注册）：
  switch<0.3（主口径=per-seed rate 跨种子 max，任一臂；次口径=mean，双口径都判）
  + F̄≤0.0331（spectral/triplet 归档基准）+ 同批配对每种子 ΔF≤0.005（vs 同批对照）。
- 判读路径（按 workflow verdict_mapping 预注册）：有候选过 gate → 选为 Phase 3 规则、
  开 experiment-long（T=50 终判）；全败 → 升级组合决策——降规模选项（k=2 规则 +
  switch_max 0.44 作限制报告跑 Phase 3）拍板落档本台账（以实测判读为准，跑前不预设）。

## [2026-10-23] F2 修复第二轮判读（experiment-redesign-2）：候选④集 7/7 未过 gate + Phase 3 规则拍板
- 网格：8 唯一单元 × 5 种子 × 5 臂 = 40 runs（results/pilot-f2fix2-metrics.json，wall 429s
  合计/单 run ~10.7s；run_f2fix2.py 驱动，可恢复）。对照 none/p1/tau2/b0.3 与 none/p2、
  none/p3 同批重跑**逐位复现轮次 1**（spectral switch 0.49/0.67、F̄ 0.0331；k=2 0.27/0.44、
  0.0333；k=3 0.13/0.22、0.0433）——对照有效 + GPU 确定性复证。
- **总判：7/7 候选 FAIL（主口径 switch_max 与次口径 switch_mean 双读都 FAIL）**。
  预注册预测对照（跑前冻结 vs 实测）：asym「switch 低于 k=2」✓（0.22<0.27）、「F̄≈k=2」✓
  （0.0304≤0.0333）、「槽位耗尽风险」✗未现（forced 0 vs k=2 的 1）；band0.5「无效或边际」
  ✓（0.47/0.67、F̄ 0.0359）；robust「告警减少、F̄ 向 baseline 回落」✓（bound 越界 13→8/40，
  F̄ 0.0444，配对 Δ+0.0113 CI95[+0.0020,+0.0242] 不含 0——显著变差）；ewma「告警局部化」
  部分✓（bound 13→8/40）但「贴线振荡」风险✓应验（seed3 switch 0.56）；detrend-bound
  「drift/beta 不直接越界」✓（drift 0/40、beta 1/40）但「bound 振荡修复」✗（0.29/0.56、
  F̄ 0.0451——动作路径/落点效应仍劣化 F）。
- **asym（④a 非对称 i2o3）最近线且严格支配 persist k=2**：switch 0.22/0.22（全种子一致，
  唯一双口径过线的候选）、F̄ 0.0304≤0.0331、isolate 15（对照 14）、forced 0（对照 2）；
  配对每种子 [-0.0100,-0.0070,-0.0080,+0.0006,**+0.0110**]——仅 seed4 ΔF=+0.0110>
  PAIR_TOL 0.005 触发 paired_not_worse=false（mean Δ=-0.0027，CI95[-0.0086,+0.0048] 含 0，
  无显著损害）。**非对称持续期打破了 persist 家族的单调 trade-off**：k=1/2/3 的
  switch 0.49/0.67→0.27/0.44→0.13/0.22 vs F̄ 0.0331→0.0333→0.0433 的单调交换被
  asym 2/3 同时改善两侧（vs k=2：switch max 0.44→0.22、F̄ 0.0333→0.0304、配对 5/5 种子
  不差于、forced 1→0）。seed4 机制：asym 进入需 2 连续告警 → 错过 t5 单尖峰、隔离块后移到
  t6-t8（对照 t5/t8/t9），早期任务遗忘抬高（t1 0.074→0.116）。
- F1 未触发（全规则跨种子汇总 max|ρ|=0.6659-0.6818<0.9）；F5 不变（cost_ratio 0.37-0.41，
  规则层修复不改变触发计算成本）。
- 管线缺陷修复（实例化缺陷，判据未动）：首版 run config 未记录 k_in/k_out/ewma_alpha
  （标签缺行为轴——轮次 1 教训复发）→ train.py 补记后同种子重生 40 runs，**行为字段
  40/40 逐位 identical**（修前/修后对比见 experiments/_probe_regen.py 输出）；分析标签
  随之带 -i2o3/-a0.3 轴。审判口径与阈值零改动。
- **拍板（组合决策，全败路径）**：Phase 3（experiment-long，T=50 终判）规则 = **asym
  （k_in=2, k_out=3, RunningZ, tau2, band0.3）**，F2 以**限制报告**携带——gate 形式上
  未过（seed4 配对 +0.0110>0.005），switch_max 0.22≤0.3 与 F̄≤0.0331 已过。理由：预注册
  降规模文本举 k=2 为例，但同批实测 asym 在每一根轴上都严格优于 k=2（见上），带 k=2 进
  终判等于带一个已知更差的规则；asym 是预注册候选④a、参数跑前冻结，选它不是事后调参。
  H1 仍未决（F3 长序列路径未走）；Phase 3 判读时 F2 限制必须与主判据结果并列报告。
- 并发记录（skill 撞车处置）：并行 wrap 收尾会话 07:32-07:56 在同一工作区改写
  STATUS/workflow.json 并提交结果（4133705/a8e6c5d/43ea954，origin 已推），其变更经
  diff 核对无语义损失（workflow.json 为驱动器回执 schema 重排）；本单元的重生结果
  （config 轴修复后）覆盖 43ea954 提交的首版——行为逐位 identical，仅补全行为轴标签。
  STATUS.md/workflow.json 为撞道热点文件，后续单元落笔前须重读+git diff 核对。
