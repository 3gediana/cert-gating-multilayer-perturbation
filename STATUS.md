# 状态

- 阶段：实验侧启动（Phase 0 完成——设计定稿 + 骨架代码 + smoke 管线验证通过；**未出任何结论**）
- 最近更新：2026-10-22（夜间生产单元，autonomous-research-production 模式）

## 产物
- `experiments/pilot-design.md`——判决性实验最小设计：五臂协议（baseline/cka/spectral/beta/triplet）、
  三量操作化定义（§3）、预注册判读标准（§6）、失败条件 F1-F5（§7）、分期计划（§8）
- `experiments/{tasks,model,triggers,gating,train}.py`——可运行骨架：
  数据准备 + slots 架构 + 三量触发（CKA 漂移 / 谱干涉界 / 锚泄漏）+ rollback-retrain 门控 + 遗忘度量
- `results/metrics.json` + `results/smoke-log.txt`——smoke 证据（五臂 × 5 任务，CUDA GPU）

## smoke 关键读数（管线验证，不作结论）
- baseline F=0.0375（acc 0.9508）；spectral/triplet F=0.0203（隔离 task 3 后回滚重训）
- 五臂参数量一致 27826；触发计算/训练耗时比 ~0.20（F5 阈值 20% 贴线——phase 1 提高 steps 后预计降至 ~0.12，盯）
- 门控因果性已验证：隔离后 trunk0 状态改变 → 后续任务三量显著偏移

## 复现入口
    PYTHONUTF8=1 G:/ai/python11/python.exe experiments/train.py --phase smoke --seed 0 \
      --out results/metrics.json --log results/smoke-log.txt

## git
- 本地仓库已建（main 分支），今晚 2+ commits（首提交 785ea69、smoke 1f2d427），未推送

## 下一步（next_units）
1. Phase 1：合成 pilot T=10 × 5 臂 × 5 种子（width 256）——主判据初判 + F1 退化检查
2. Phase 2：Split-CIFAR10 子集（5 任务 × 2 类）3 种子——分布迁移验证
3. Phase 3：合成 T=50 长序列——"长序列才显著"预测验证

## 阻塞
- 无。F1 三量退化检查待 Phase 1 数据（smoke T=5 只有 4 个触发样本，相关性不可判读）
