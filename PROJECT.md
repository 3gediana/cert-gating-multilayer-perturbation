<!-- dossier:seed:start -->
# 项目档案｜证书门控-多层扰动度量

- 立项日期：2026-10-21
- 立项理由：组合多种指标做门控，如果只是「三层比一层好」，在评审眼里等于没有贡献：组合型贡献必须论证每层对应一条独立失效通道（notes/DECISIONS.md 2026-10-21 taste 条）。所以这个项目先论证三条通道互补，再用实验检验三元组是否真的比任一单量多带来决策增量。
- 目的：回答「持续学习的结构决策该用什么触发量」：三元组相对单量有没有额外价值、多花多少成本；即使答案是负的，也要留下可复现的判断依据。
- 研究问题：持续学习的结构决策（并入共享结构/隔离/扩展容量/重算全局结构）的触发量应是三元组——表征层 CKA 稳定/可塑性（S_i，漂移 d=1-S）、更新层谱干涉界 ||dW||_2·||J||_*、几何层锚泄漏 beta_{s,r}——而非任一单量？完备性论证：三层是三条独立失效通道（表征几何改写 / 最坏情况一阶干涉上界 / 更新子空间与锚梯度方向共线），见 experiments/pilot-design.md §0。
- 可证伪预测：同一协议、等参数量、同记忆预算下，三元组门控（A4）平均遗忘低于任一单量门控（A1 cka / A2 spectral / A3 beta），配对种子 bootstrap CI95 不含 0 且 delta-F>=5pp（主判据）；增益在长序列（T=50 vs T=10）上放大（次判据②）；三元组相对不门控 baseline 有效（次判据①）。当前读数（截至 2026-10-22）：次判据①在合成 pilot 成立（-41%，5/5 种子）但在图像上不保持；主判据在 T=10 与 Split-CIFAR10 均未达（F3 路径延续，Phase 3 终判）。
- 失败条件：预注册于 experiments/pilot-design.md §7（本 workflow 只引用，不得事后修改）：F1 任一臂内三量两两 |rho|>0.9 → 三元组退化为单量，H 驳倒；F2 任一臂 action-switch rate>30%（hysteresis 已开）→ 阈值规则失效，重设计后再判；F3 等参数量下三元组不显著优于任一单量 → 升级长序列再判，仍无则驳倒；F4 工程失败（NaN/CKA 未定义/SVD 失败）→ 修管线并记 FAILURE-PATTERNS；F5 触发计算>单任务训练 20% → 可扩展性缺陷，必须报告。已触发与处置见 notes/FAILURE-PATTERNS.md。
- 最小可判别实验：Phase 3 长序列终判：合成 T=50 × 5 臂 × 5 种子（F2 修复后规则），判 triplet vs 每一单量臂 delta-F̄（配对 bootstrap CI95 不含 0 且 >=5pp）+ 次判据②（T=50 vs T=10 增益放大）+ F1 跨种子检查（F1 判读必须跨种子汇总，单种子 n 小会假高）。
- 读者能学到：怎么把一个「组合型」研究问题拆成可证伪的失败条件，以及为什么主判据要事先写死、负结果也要原样保留。
<!-- dossier:seed:end -->

## 当前快照（系统按项目事实维护）
- 当前阶段：experiment-long（Phase 3 终判）完成——H1 在合成尺度 falsified（F3 终判，负结果）：T=50 × 5 臂 × 5 种子，triplet vs spectral ΔF̄=0（动作 5/5 全同）、vs cka/beta +0.0374（CI95 不含 0 但 <5pp）→ 主判据 FAIL；F1 未触发（0.7331<0.9）、F2 T=50 未触发（0.04；T=10 gate 未过状态以限制报告并列）、F5 严重触发（cost_ratio 3.10，成本曲线已附）。详见「Phase 3 终判读数」节。
- 下一步：experiment-long（Phase 3 终判）已完成（2026-10-23 09:2x）：H1 合成尺度终判 falsified （F3，负结果）——25 runs（T=50 × 5 臂 × 5 种子，规则=asym），主判据 FAIL（vs spectral
- 阻塞：无硬阻塞。H1 已在合成尺度终判 falsified（F3）：三元组门控相对单量（bound/spectral） 无决策增量（T=10 与 T=50 均为 ΔF̄=0，动作逐位一致），对 cka/beta/baseline 的增益全部来自
- 最近结果：判定 falsified；发现 FAIL：triplet vs spectral ΔF̄=0.0000（动作 5/5 种子逐位一致）；vs cka ΔF̄=+0.0374 CI95[+0.0299,+0.0439] 不含 0 但 <5pp；vs beta 同幅 <5pp；vs baseline +0.0374 <5pp（次判据①合成成立但幅度不足）
- 关键指标：{"runs": 25, "F_triplet": 0.2928, "F_spectral": 0.2928, "F_baseline": 0.3302, "f1_max_abs_pooled_rho": 0.7331, "f5_cost_ratio": "0.38（T=10）→1.19（T=25 探针）→3.10（T=50）（阈值 0.2，严重触发）"}

## 决策要点（最近记录）
- triplet vs spectral：ΔF̄=0.0000（动作序列 5/5 种子完全一致）——CI 不含 0 否、5pp 否；
- triplet vs cka：ΔF̄=+0.0374，CI95[+0.0299,+0.0439] 不含 0、5/5 种子同向，但 <5pp；
- triplet vs beta：同 cka（+0.0374，CI 同上，<5pp）；
- （vs baseline 并列：+0.0374，CI 不含 0，<5pp——次判据①在合成上仍成立但幅度不足。）
- 主判据总判 FAIL（要求三个单量臂全部 CI95 不含 0 且 ≥5pp）。
- 结构解释（新事实）：T=50 下 triplet 的全部增益来自 bound 单通道——spectral 与
- 次判据②（长序列增益放大）：ΔF̄(triplet−单量) T=10→T=50：vs cka 0.0226→0.0374
- F1：未触发。跨种子汇总（n=245/臂/对）max|ρ|=0.7331（drift~bound，spectral/triplet
- F2：T=50 未触发（switch mean/max=0.04/0.04，spectral/triplet；其余臂 0）。
- F5：严重触发。cost_ratio 0.38（T=10）→1.19（T=25）→3.10（T=50）（阈值 0.2）。
- 管线步骤：train.py --T 覆盖（smoke T=4 验证）；产物 results/phase3-metrics.json +
- 结论落档：H1 在合成尺度被驳倒（falsified，F3 终判）——三元组门控相对单量

## 给讲解者的说明
- 以上是项目全部已确认事实；没有写的事实不存在，不要猜数字或结论。
- 立项理由和目的是项目负责人拍板留下的，不要改写或发挥。
