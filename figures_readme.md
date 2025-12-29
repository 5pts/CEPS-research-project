# 可视化图表说明文档

本文件夹包含本研究的所有可视化分析图表，分为**基础描述性统计**和**机器学习探索性分析**两部分。

## 1. 基础描述性统计 (位于 `figures/` 根目录)

展示学校社会资本与教育期望的总体关系及调节效应。

### 1.1 主效应检验
*   **bonding_vs_expect_prob.png** (师生关系 vs 升学信心)
    *   *内容*：将 Bonding SC 按分位数分组，展示每组学生的平均升学信心概率。
    *   *预期*：随 Bonding SC 增加，升学概率上升。
*   **bridging_vs_expect_prob.png** (同伴氛围 vs 升学信心)
    *   *内容*：将 Bridging SC 按分位数分组，展示每组学生的平均升学信心概率。
    *   *预期*：随 Bridging SC 增加，升学概率上升。

### 1.2 调节效应检验 (家庭背景)
*   **bonding_expect_by_SES.png** (SES 的调节作用)
    *   *内容*：将学生分为高/低 SES 两组，分别绘制 Bonding SC 与升学信心的关系。
    *   *补偿效应验证*：若**低 SES 组**的线条斜率比高 SES 组更陡峭，说明师生关系对弱势学生更重要（支持假设 H2a）。
*   **rescued_interaction_hukou.png** (户籍的调节作用，位于 `rescued_figures/`)
    *   *内容*：将学生分为城市/农村户籍两组，比较 Bonding SC 与升学信心的关系。
    *   *补偿效应验证*：若**农村户籍组**的线条斜率更陡峭，说明师生关系对农村学生有更强的补偿作用（支持假设 H2c）。

---

## 2. 机器学习探索性分析 (位于 `figures/ml_insights/` 目录)

使用广义加性模型 (GAM) 和决策树挖掘非线性关系与阈值。

### 2.1 非线性效应 (GAM)
GAM 模型形式：$$ g(E[Y]) = \beta_0 + f_1(Bonding\_SC) + f_2(Bridging\_SC) + \beta_3(SES) + \beta_4(Hukou) $$

*   **GAM_Bonding_Partial_Dependence.png**
    *   *横轴*：Bonding SC 指数（Z-score）。
    *   *纵轴*：对升学信心的边际贡献 (Log Odds)。
    *   *洞察*：观察曲线形态。是线性的？还是在低分段陡峭、高分段平缓（回报递减）？
*   **GAM_Bridging_Partial_Dependence.png**
    *   *洞察*：观察是否存在“临界点”，即同伴质量必须达到一定水平才能起效。

### 2.2 阈值识别 (Decision Tree)
*   **Decision_Tree_Thresholds.png**
    *   使用浅层决策树（Depth=3）自动寻找区分高/低期望的最优切分点。
    *   *解读*：树的根节点（Root Node）指出了全样本中最重要的区分指标和阈值（例如 `bonding_sc_idx <= -0.5`）。

### 2.3 交互效应平滑曲线
*   **Interaction_Bonding_SES.png**
    *   使用多项式拟合展示不同 SES 组别的趋势差异，比基础统计图更平滑，能展示非线性交互（例如：在低 Bonding 区域差异不大，但在高 Bonding 区域差异拉大）。

---
**变量说明**：
*   **Bonding SC**: 师生关系指数（标准化）。
*   **Bridging SC**: 同伴氛围指数（标准化）。
*   **SES**: 家庭社会经济地位指数（标准化）。
*   **Expectation**: 升学信心（1=期望上普高/大学，0=否）。
