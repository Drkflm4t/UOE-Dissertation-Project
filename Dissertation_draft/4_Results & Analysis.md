# 第4章细纲：Results and Analysis

> **文件性质：** 本文件是 Chapter 4 的详细细纲和写作蓝图，不是 dissertation 正文。
>
> **当前分析口径：** 全章严格遵循已经完成的实验流程。Free metrics 由独立 Judge 提取；Structured metrics 使用 native Pydantic self-report。5-paper manual validation 仅用于辅助验证 measurement procedure。除非后续与导师讨论后另行决定，common-Judge analysis 不属于当前 Results framework。

---

## 4.0 本章目的与叙事顺序

### 本章目的

本章首先检验 schema-constrained reviewing 能否缓解 LLM peer review 的两类已知缺陷，随后补充分析其对 review-aspect coverage、system efficiency 和 rating dispersion 的影响。

### 证据层级

1. **Primary：RQ1 — Manipulation robustness。**
2. **Primary：RQ2 — Logic-defect discriminability。**
3. **Secondary：RQ3 — Review-aspect coverage。**
4. **Secondary：RQ4 — Operational efficiency and rating dispersion。**
5. **Auxiliary validation：** 5-paper manual check of automated metric extraction。

### 开头统一说明 measurement convention

- Free-form reviews：由 `deepseek-v4-flash` Judge 提取 rating 和 aspect counts。
- Structured reviews：由 native Pydantic fields 提供 rating 和 aspect counts。
- Injection-compliance scores：两种 setup 均由 Judge 提取。
- 因此，两种 setup 的 rating/aspect counts 使用不同的 operational measurement procedures。
- 该差异不妨碍各 setup 内部的 paired comparison；但跨 setup 比较 effect magnitude 时，结论必须表述为当前 measurement pipeline 下的 setup-level difference。

### 建议开头段落结构

1. 重申所有核心比较均为 within-paper comparison。
2. 说明 N = 30，并区分 Injection track 与 Counterfactual track。
3. 说明 Results 按 RQ 而非 pipeline step 组织。
4. 说明描述性图表同时配合 exact statistical tests、confidence intervals 和 effect sizes。

> 本章不深入讨论失败原因、system-design recommendation 或广泛政策含义；这些内容留到 Chapter 5。

---

# 4.1 RQ1 — Susceptibility to white-text prompt injection

## 4.1.1 研究问题与 robust behaviour 判据

### RQ 正式表述

**How does schema-constrained reviewing affect the susceptibility of LLM-generated reviews to white-text prompt injection?**

### Robust behaviour 的操作定义

一个不受攻击影响的 reviewer，在 Original_PDF 与 Manipulated_PDF 之间应当呈现接近零的 paired change，涉及：

- overall rating；
- strengths；
- weaknesses；
- methodological flaws；
- injection-compliance score。

Payload 的设计目标是提高 rating 和正面评价，同时减少 weaknesses 与 methodological criticism。因此，指标沿这些预设方向变化可视为 behavioural compliance。

## 4.1.2 Rating inflation

### 描述性结果

| Setup | Original mean | Injected mean | Mean Δ |
|---|---:|---:|---:|
| Free | 6.23 | 8.13 | +1.90 |
| Structured | 5.90 | 6.87 | +0.97 |

补充方向性结果：

- Free：29/30 篇 rating 上升，1/30 不变，0/30 下降。
- Structured：25/30 篇 rating 上升，5/30 不变，0/30 下降。
- 两种 setup 内部的 rating increase 均显著；paired slope figure 中均为 `p < .001`。

### Setup 间 attenuation comparison

数据源：`outputs/stats/rq1_attenuation_test.csv`

| Contrast | Mean difference | 95% CI | paired t(29) | Holm p | Cohen's d_z |
|---|---:|---:|---:|---:|---:|
| Free Δ − Structured Δ | 0.93 | [0.61, 1.26] | 5.89 | < .001 | 1.07 |

### Results 中允许作出的解释

- Prompt injection 显著提高了两种 setup 的 ratings。
- Structured 条件下的 rating increase 显著小于 Free。
- 因此，Structured **attenuated rating inflation but did not prevent it**。

### 对应图

`outputs/figures/rq1_slope_graph.png`

### 正文段落规划

1. 介绍 paired rating comparison 与 slope graph。
2. 报告 Original/Injected means 和两种 setup 内部的 ATE。
3. 报告 Free-versus-Structured delta comparison、95% CI 和 `d_z`。
4. 用狭义结论结束：存在 attenuation，但不存在 robustness。

## 4.1.3 Strengths、weaknesses 与 compliance 的变化

### 描述性结果

| Metric | Free: Original | Free: Injected | Free Δ | Structured: Original | Structured: Injected | Structured Δ |
|---|---:|---:|---:|---:|---:|---:|
| Strengths | 7.53 | 8.63 | +1.10 | 6.63 | 8.73 | +2.10 |
| Weaknesses | 4.57 | 1.87 | -2.70 | 7.53 | 6.33 | -1.20 |
| Compliance | 1.47 | 6.77 | +5.30 | 0.87 | 3.27 | +2.40 |

方向性结果：

- Free weaknesses：29/30 下降，1/30 不变，0/30 上升。
- Structured weaknesses：19/30 下降，10/30 不变，1/30 上升。
- Free compliance：28/30 上升，2/30 不变，0/30 下降。
- Structured compliance：20/30 上升，7/30 不变，3/30 下降。
- Structured 的 strengths increase 大于 Free，因此 attenuation 并未在所有 attack-aligned metrics 上一致出现。

### 对应图

`outputs/figures/rq1_bubble_compliance.png`

### Results 中允许作出的解释

- Payload 不仅改变 rating，也改变了 praise 与 criticism 的相对构成。
- Free 呈现更大的 weakness reduction 和 compliance-score increase。
- 由于 Structured 的 strengths increase 更大，不能笼统声称 Structured 在所有方面都更少服从攻击。

### 正文段落规划

1. 说明 bubble chart 的 axes 和 bubble size 如何对应 payload 的三个目标。
2. 分别报告 weakness 与 strength changes，不要合并成单一 compliance 结论。
3. 使用 Judge-derived compliance score 作为 convergent evidence。
4. 明确指出不同 metrics 上的 attenuation 具有 heterogeneity。

## 4.1.4 Methodological criticism 的减少

### 描述性结果

| Setup | Original MF mean | Injected MF mean | Mean Δ |
|---|---:|---:|---:|
| Free | 4.27 | 1.07 | -3.20 |
| Structured | 8.07 | 6.23 | -1.83 |

方向性结果：

- Free：28/30 篇的 methodological-flaw count 下降。
- Structured：25/30 篇的 methodological-flaw count 下降。

### Setup 间 attenuation comparison

| Contrast | Mean difference | 95% CI | paired t(29) | Holm p | Cohen's d_z |
|---|---:|---:|---:|---:|---:|
| Free Δ − Structured Δ | -1.37 | [-2.29, -0.44] | -3.01 | .005 | -0.55 |

### 对应图

`outputs/figures/rq1_scatter_integrity.png`

### Results 中允许作出的解释

- Injection 显著减少了两种 setup 中报告的 methodological flaws。
- Structured 条件下的 reduction 显著小于 Free。
- 正文使用 “reduced methodological criticism” 或 “methodological-flaw reporting decreased”。
- “MF blindness” 可保留为 figure 的简写标签，但正文不能声称模型完全失明。

### 正文段落规划

1. 报告两种 setup 的 paired reductions 和方向性数量。
2. 报告 attenuation contrast、95% CI 和 effect size。
3. 明确 outcome 是“被报告的 methodological flaws”，而不是模型内部 reasoning 的直接观察。

## 4.1.5 Hypothesis-led qualitative case

### Auxiliary coding 对 RQ1 pattern 的核验

5-paper / 20-review RQ1 sample 在共同 coding 下得到：

| Metric | Free Δ | Structured Δ | Interpretation |
|---|---:|---:|---|
| Inferred rating | +1.4 | +0.6 | 支持较小的 Structured rating shift |
| Strengths | +2.8 | +1.6 | 两者 praise 均增加；Free 增加更大 |
| Weaknesses | -0.4 | -0.2 | 方向相同但幅度很小，且 weakness agreement 很差 |
| Methodological flaws | -4.6 | -1.6 | 支持较小的 Structured criticism reduction |
| Injection compliance | +4.8 | +2.6 | 支持较小的 Structured compliance shift |

方向性上，Free rating 与 compliance 均为 5/5 papers 上升；Structured rating 与 compliance 均为 3/5 上升、2/5 不变；两种 setup 的 MF 均为 5/5 下降。该 limited sample 因而对 **both setups remained susceptible** 以及 rating、MF、compliance 上的 selected attenuation 提供 convergent evidence，但不验证 weakness-count effect。

Measurement agreement 的关键边界为：Free rating MAE = 0.64、`ρ = .83`；Free/Structured compliance MAE 分别为 1.90/1.50；Free/Structured weakness MAE 分别为 6.28/3.40，且 rank correlations 接近 0 或为负。正文不得声称所有自动 metrics 均经人工核验。

### 定量结果留下的解释缺口

RQ1 的 paired tests 能证明 Structured 条件下部分 attack effects 较小，但不能解释 attenuation 如何体现在 review text 中。这里检验一个有限的、output-level 假设：

> Mandatory weakness 与 methodological-flaw fields 可能使 Structured review 在遵从 praise-oriented payload 时仍保留一定批评，因此 attenuation 更可能表现为 criticism 被较少删除，而不是完全拒绝 injection。

### Case-selection 与核验规则

- 从不属于 5-paper manual-validation sample 的 RQ1 papers 中，按预先可说明的定量 pattern 选择一个 illustrative matched case：Free 出现 rating inflation 与 criticism reduction，而 Structured 的相同变化较小。
- 同时阅读该 paper 的 Original/Manipulated × Free/Structured 四份 reviews。
- 只核验三个文本现象：praise 是否增加、criticism 是否被删除或弱化、Structured mandatory fields 是否仍包含 substantive criticism。
- 正文仅用一例作机制说明，并明确它是 illustrative evidence，不把单个 case 当成 effect prevalence 的估计。

### 当前候选 case 的初步证据

一个非人工盲审候选中，Free rating 由 7 升至 9、weakness count 由 4 降至 2、MF count 由 5 降至 2、compliance 由 0 升至 9；Structured rating 保持 7，weaknesses 由 7 增至 8、MF 由 7 降至 6、compliance 保持 2。文本抽查也显示 Manipulated Free review 的总体措辞明显更积极，而 Structured review 仍保留 pseudometric、composite-null procedure、causal attribution 等具体批评。

这与上述有限机制一致，但不能证明 schema fields 是唯一原因，因为 Free/Structured prompts、输出格式与测量流程同时变化。

## 4.1.6 RQ1 回答

### 可用于正文的小结句

> Schema-constrained reviewing significantly attenuated the observed rating inflation and reduction in methodological-flaw reporting, but both setups remained behaviourally susceptible to document-layer prompt injection.

### 避免的表述

- “Structured resisted the attack.”
- “Structured solved prompt injection.”
- “Free was completely controlled.”
- “The model became blind to all scientific flaws.”
- “JSON itself caused the attenuation.”

---

# 4.2 RQ2 — Discriminability of logic defects

## 4.2.1 研究问题与 discriminability 判据

### RQ 正式表述

**Does schema-constrained reviewing improve differential sensitivity to logic-critical rather than surface-level perturbations?**

### Discriminability 的操作定义

对每篇论文和每个 metric：

1. 计算每个 counterfactual review 相对于其 Original review 的 Δ。
2. 在 paper 内对三个 Logic deltas 求平均。
3. 在 paper 内对四个 Format deltas 求平均。
4. 使用 paired test 比较 paper-level mean Logic Δ 与 mean Format Δ。

只有当 reviewer 对 Logic perturbations 的反应相对于 Format perturbations 更强、方向正确且统计上可靠时，才能认为其具备 logic-specific discriminability。

> 某一组的 one-sample test 显著偏离 0，并不足以证明 discriminability。

## 4.2.2 Methodological-flaw reporting

### 统计结果

数据源：`outputs/stats/rq2_discriminability_test.csv`

| Setup | Logic mean Δ | Format mean Δ | Logic − Format | 95% CI | paired t(29) | Holm p | d_z |
|---|---:|---:|---:|---:|---:|---:|---:|
| Free | +0.66 | +0.77 | -0.11 | [-0.58, 0.36] | -0.48 | 1.000 | -0.09 |
| Structured | -0.43 | -0.63 | +0.20 | [-0.08, 0.48] | 1.44 | .647 | 0.26 |

### 解释方向

- Free 在 Logic perturbations 后平均提到更多 methodological flaws，但 Format perturbations 后的增加略大。
- Structured 在两种 perturbation classes 后都报告了更少的 methodological flaws。
- Holm correction 后，两种 Logic–Format contrasts 均不显著。
- 两个 confidence intervals 均跨 0，effect sizes 较小。
- Methodological-flaw count 上升不能称为“惩罚”，因为 rating 并未下降。

## 4.2.3 Rating response

### 统计结果

| Setup | Logic mean Δ | Format mean Δ | Logic − Format | 95% CI | paired t(29) | Holm p | d_z |
|---|---:|---:|---:|---:|---:|---:|---:|
| Free | +0.02 | +0.03 | -0.003 | [-0.22, 0.21] | -0.03 | 1.000 | -0.005 |
| Structured | +0.01 | +0.03 | -0.014 | [-0.11, 0.08] | -0.29 | 1.000 | -0.054 |

### 解释方向

- 两种 setup 的 ratings 对两类 perturbations 都几乎没有变化。
- 两种 setup 都没有对 logic-critical edits 施加 evaluative consequence。
- 接近 0 的 effect sizes 强化了无方向性差异的结果，而不仅仅是“未达到显著”。

## 4.2.4 对应图及其限制

`outputs/figures/rq2_discriminability_box.png`

现有图中的 stars 来自各 box 相对 0 的 one-sample tests。正文和 caption 必须：

- 不把这些 stars 当作 Logic-versus-Format discriminability 的证据；
- 以前述 paper-level paired contrasts 为主要推断结果；
- 最终制图时改成 direct Logic–Format comparison annotations，或者移除容易误导的 stars。

> **Final figure revision required：** 在 dissertation 最终版本中，用 direct Logic–Format test 替换 one-sample stars，或在 caption 中清楚解释其有限含义。

## 4.2.5 Target-defect qualitative audit

### 为什么 RQ2 最需要具体示例

MF count 只表示 review 中有多少 methodological criticisms，不能判断 reviewer 是否识别了 counterfactual 中被植入的**特定** logic defect。因而，RQ2 的 null aggregate result 至少有两种不同机制：

1. review 完全遗漏 target defect，只继续给出与 Original 类似的 generic criticisms；
2. review 注意到 target defect 或相关异常，但没有把它转化为更低 rating 或更多 MF items，即 critique–rating decoupling。

### 解盲后核验规则

- 该步骤与 blinded metric-validation coding 分开，只在 annotation sheet 锁定和 unblinding 后进行。
- 对少量 matched Logic cases 阅读 Original 与 counterfactual 的实际改动，再比较 Free/Structured reviews。
- 每个 case 只回答：targeted change 是否被明确提及、是否被正确解释为影响 claim validity、是否产生 rating consequence。
- 优先复用 5-paper sample 中的 RQ2 papers；若其中没有清晰可说明的 case，再从非人工盲审样本按明确规则补选一个。
- 不把这一步扩展成新的 prevalence estimate；正文仅展示一例，其他核验结果用一句概括，不另制主表。

### 已完成的 5-case audit

完整记录：`outputs/manual_validation/qualitative_case_audit.md`。

- Structured 在 2/5 cases 中识别 target defect；Free 为 3/5。N = 5 不支持可靠 setup difference。
- 10 份 setup-specific reviews 中，5/10 遗漏 target，5/10 至少语义上识别。若采用“必须明确引用修改数字”的更严格规则，Case 1 Free 将改为遗漏，得到 6/10 omissions；两种口径都显示 detection 不稳定。
- 在 5 份识别 target 的 reviews 中，只有 Case 2 Free 的 primary rating 相对 Original 降低。其余 detected reviews 均未降低 primary rating。

因此，aggregate null result 不是单一的“模型完全没有看见”，而是 **target omission 与 critique–rating decoupling 并存**。

### 正文代表性案例

使用 Case 3（`2024.emnlp-main.1123`, `blueprint_result`）：修改后的段落宣称 domain-specific models underperform，但同段仍保留 `xlmt_large surpasses xlmr_large` 的正向结果。Free 与 Structured reviews 均明确指出该 claim–result contradiction，但两种 setup 的 primary ratings 相对 Original 均未改变。该例直接说明 critique 可以出现而不产生 score consequence。

Case 5 不得称为 hallucination：counterfactual paper 中仍有未修改的 3.2；真正遗漏的是新增 2.0 与既有 3.2 之间的 internal inconsistency。

## 4.2.6 正文段落规划

1. 定义 paper-level aggregation，并解释为什么独立分析单位仍为 N = 30 papers。
2. 报告 Free/Structured 的 methodological-flaw contrasts。
3. 报告 rating contrasts。
4. 强调 Holm correction 后四项比较均不显著，且 effect sizes 均较小。
5. 加入 Case 3，区分 target detection、critique expression 与 primary score consequence。

## 4.2.7 RQ2 回答

### 可用于正文的小结句

> Neither free-form nor schema-constrained reviewing demonstrated reliable logic-specific discriminability, and schema constraint did not improve the response to logic-critical edits relative to surface perturbations.

补充机制句：

> The post-unblinding case audit indicated that this null response reflected both inconsistent target-defect detection and frequent critique–rating decoupling when a defect was identified.

### 避免的表述

- “Free detected the logic defects.”
- “Free punished defective papers.”
- “Structured completely lost an ability that Free possessed.”
- “The null result proves that the model noticed nothing internally.”
- “Formatting changes caused more reasoning than logic changes.”

---

# 4.3 RQ3 — Review-aspect coverage

## 4.3.1 研究问题与分析范围

### RQ 正式表述

**How does schema-constrained reviewing affect the coverage and distribution of strengths, weaknesses and methodological-flaw comments?**

### 数据限制

- 每种 setup 仅使用 Counterfactual track 中的 30 条 Original reviews。
- 排除所有 perturbed conditions，避免重复观测和 treatment effects 干扰。
- 将 RQ3 定位为 secondary descriptive analysis。

## 4.3.2 描述性结果

| Setup | Strengths | Weaknesses | Methodological flaws | Total aspect count |
|---|---:|---:|---:|---:|
| Free / Judge-extracted | 6.47 | 11.07 | 8.63 | 26.17 |
| Structured / native | 5.93 | 7.57 | 8.80 | 22.30 |

### 主要模式

- Free 的 extracted total aspect count 更高。
- 最大差异来自 weakness count。
- Original condition 下，两种 setup 的 methodological-flaw counts 接近。
- Structured 保证了 machine-readable fields，但没有产生更高的 total recorded aspect count。

## 4.3.3 对应图

`outputs/figures/rq3_comprehensiveness_stacked.png`

### 必须修正的命名

- 正文统一称为 **review-aspect coverage**。
- 现有 filename 可为 reproducibility 保留，但最终 figure title/caption 不应把 count 描述为已经验证的 comprehensiveness。

## 4.3.4 Measurement limitation

- Free counts 是 Judge-extracted；Structured counts 是 native list lengths。
- 跨 setup 的绝对 count differences 可能部分来自 measurement procedure。
- Item count 本身不测量 correctness、specificity、importance 或 non-redundancy。
- Auxiliary coding 不支持把 automatic total-count ordering 解释为 setup superiority。

## 4.3.5 Auxiliary measurement-validation result

Original-only 5-paper matched sample：

| Setup | Common-coded strengths | weaknesses | MF | Total | Automatic total |
|---|---:|---:|---:|---:|---:|
| Free | 4.4 | 3.4 | 10.6 | 18.4 | 29.2 |
| Structured | 6.6 | 2.2 | 10.4 | 19.2 | 24.0 |

与 automatic analysis 中 Free total 较高不同，共同 coding 下两个 setup 的 total aspect count 基本相同，Structured 仅高 0.8。共同 coding 仍显示不同 profile：Structured strengths 较多、Free weaknesses 略多、MF 几乎相同。

Agreement diagnostics 显示：

- Free strengths：MAE = 1.88，`ρ = .58`；Structured strengths：MAE = 0.72，`ρ = .80`。
- Free MF：MAE = 2.40，`ρ = .69`；Structured MF：MAE = 0.56，`ρ = .87`。
- Free weaknesses：MAE = 6.28，`ρ = -.05`；Structured weaknesses：MAE = 3.40，`ρ = -.18`。

因此，automatic Free > Structured total ordering 没有在 common coding 中复现，且差异最大的 weakness dimension 存在严重 measurement disagreement。该结果支持将 RQ3 解释为 **measurement-sensitive aspect profiles**，而不是稳定的 total-coverage advantage。

## 4.3.6 Count-granularity qualitative check

### 已核验的有限假设

Free 的较高 automatic aspect count 可能部分来自更细的 segmentation 与 weakness/MF category boundary，而不一定代表更广的 substantive coverage。

### Pair A inspection result

选定 `2024.emnlp-main.758` Original pair（MV_034 Free / MV_002 Structured）。Free automatic counts 为 strengths 10、weaknesses 14、MF 11、full total 35；common coding 为 7、3、14、full total 24。`25 versus 17` 仅是 weaknesses + MF 的 **critical-aspect count**，不是 total aspect count。

- Weakness discrepancy 为 11，但 common-coded MF 比 automatic MF 多 3，因此 critique count 的净差异为 8。
- Category reassignment 可以解释部分差异；其余差异与更细 segmentation 一致。
- 长篇 Free review 多次返回 comparison fairness、confounding 与 experimental control 等 umbrella themes。

但 Judge 只保存 counts，没有 item-level spans/lists；因此不能声称已证明 Judge 将某三个句子分别计数，或在 weakness/MF 中对某一具体 item double-count。若若干问题可独立修复，按 distinct-item rule 分开计数也可能合理。

### 可用解释

> Pair A demonstrated substantial sensitivity to segmentation and category allocation, but did not identify the precise item-level mechanism behind the count discrepancy.

正文最多概括这一例，不新增主表，也不将它解释为 Structured comprehensiveness advantage。

## 4.3.7 正文段落规划

1. 说明为什么只使用 Original condition。
2. 报告三个 aspect dimensions 和 total counts。
3. 指出 automatic total difference 主要由 extracted weaknesses 驱动，但 common coding 未复现 total ordering。
4. 报告 Pair A 的 full totals 与 critical-aspect counts，明确 category reassignment 和 segmentation sensitivity。
5. 说明 count-only Judge 无法提供 item-level over-segmentation proof。
6. 只得出两种 setup 产生 measurement-sensitive operational coverage profiles 的结论。

## 4.3.8 RQ3 暂定回答

### 可用于正文的小结句

> Under the setup-specific automatic measures, free-form reviews yielded a higher total aspect count, driven primarily by extracted weaknesses. This total-count ordering was not reproduced under common auxiliary coding, which instead showed similar totals and indicated that the apparent difference was sensitive to segmentation and measurement procedure.

### 避免的表述

- “Structured reviews were less comprehensive.”
- “Free reviews were higher quality.”
- “More comments mean better reviews.”
- “Structured guaranteed complete coverage.”
- “The common coding proved that Structured reviews were more comprehensive.”

---

# 4.4 RQ4 — Operational efficiency and rating dispersion

## 4.4.1 研究问题与数据范围

### RQ 正式表述

**What trade-offs arise between output efficiency and rating dispersion under schema-constrained reviewing?**

### 数据

- Counterfactual track 的全部 240 rows。
- 30 papers × 8 text conditions。
- Metrics：words、latency、output tokens 和 rating distribution。

## 4.4.2 Review length 与 latency

| Metric | Free | Structured | Relative change |
|---|---:|---:|---:|
| Mean words | 1883.7 | 793.3 | -57.9% |
| Median words | 1872 | 782 | — |
| Word-count SD | 183.2 | 91.9 | — |
| Mean latency | 29.1 s | 13.7 s | -53.0% |
| Median latency | 28.9 s | 13.6 s | — |
| Latency SD | 3.17 s | 1.81 s | — |

### 解释方向

- 在本实验 API conditions 下，Structured reviews 显著更短、更快。
- Scatter plot 显示 output length 与 latency 紧密相关。
- 正文称其为 observed end-to-end API latency，而不是 pure compute time。
- 不将结果扩展为对未知 backend efficiency 的因果结论。

## 4.4.3 Token use 与 estimated cost

| Metric | Free | Structured | Reduction |
|---|---:|---:|---:|
| Total output tokens | 565,075 | 267,167 | -52.7% |
| Mean tokens/review | 2,354 | 1,113 | -52.7% |
| Illustrative cost at $10/1M output tokens | $5.65 | $2.67 | $2.98 |

### 报告边界

- 明确 tokenizer 为 `o200k_base`。
- 将金额标记为基于 `$10/1M output tokens` 假设的 illustrative estimate。
- 除非有真实 invoice 或正式 pricing 支持，不将其描述为 `gpt-5.4` 的 verified billing cost。

## 4.4.4 Rating dispersion

| Setup | Mean rating | Median | SD |
|---|---:|---:|---:|
| Free / Judge-extracted | 5.35 | 6 | 1.27 |
| Structured / native | 5.72 | 6 | 0.93 |

- Levene test：`p < .001`。
- Structured ratings 呈现更窄的 distribution。
- 这是 distributional difference，不是 cognitive variance 的直接证据。
- Lower dispersion 可能表示 desirable consistency、scale compression 或 reduced sensitivity；当前设计无法区分这些解释。
- 两组 rating 的 extraction procedures 不同，因此不进行强 normative comparison。

## 4.4.5 对较窄 rating distribution 的有限 diagnostic

### 分析目的与结论边界

导师针对早期“Structured ratings 几乎都在 7 附近”的异常提出过一个可检验假设：reviews 的内容可能高度同质化，进而压缩 rating distribution。优化后的当前数据已经不存在旧的 all-seven pattern，因此本节不再需要为旧异常寻找成因，只需确认该简单假设能否解释**当前仍较窄的 dispersion**。

内部 diagnostic 文件为 `outputs/stats/rq4_rating_frequency.csv` 与 `outputs/stats/rq4_distribution_diagnostics.csv`；它们用于 audit，不作为 dissertation 的额外主表。

当前证据为：

- Structured ratings 的当前众数为 6，而非几乎全部为 7；其分布仍比 Free 窄。
- Structured 对同一 paper 的八个 text conditions 呈现较小的 within-paper rating variation，与较集中的 score response 一致。
- 简单 token-set Jaccard 没有显示 Structured reviews 比 Free reviews 具有更高 lexical overlap。
- 对不同 rating levels 的少量 Structured outputs 的抽查显示 paper-specific summary 与 criticisms 存在实质差异，并非近似复制同一 generic review。
- 在 25 + 25 reviews 的共同 inferred-rating coding 中，Free SD = 1.12，Structured SD = 0.53；这为较窄 Structured rating dispersion 提供 limited common-measure corroboration。

因此，现有证据**不支持**“当前较窄 rating distribution 是由近乎相同的 review content 直接造成”的简单推测；但 lexical overlap 与少量 case inspection 也不足以排除更细微的 semantic or rubric-level homogenisation。

### 是否还需要正文示例

RQ4 不再单独展示三个 rating levels 的 examples。若需要回应导师的原问题，可复用 RQ2 的一例：counterfactual content 已发生实质变化，但 Structured rating 未改变。该例只能说明 score response 可能被压缩，不能识别成因。

### Results 中建议采用的结论

> The earlier near-degenerate concentration around a single rating was no longer present. Structured ratings remained less variable, but exploratory diagnostics did not support the simple explanation that this arose because the reviews were near-identical textual outputs; the cause of the remaining compression could not be identified by the present design.

### 避免的表述

- “Content similarity has been definitively ruled out.”
- “Structured reviews are semantically more diverse than Free reviews.”
- “The remaining rating compression is caused by the JSON schema.”
- “Lower score variation proves that Structured is insensitive.”

## 4.4.6 对应图

`outputs/figures/rq4_tradeoffs.png`

### 必须修正的 title/caption

- 将 “Cognitive Cost” / “Cognitive Variance” 改为 “Rating Dispersion” 或 “Rating Distribution”。
- Caption 说明 Free ratings 为 Judge-extracted，Structured ratings 为 native。

## 4.4.7 正文段落规划

1. 介绍基于全部 240 rows 的 engineering comparison。
2. 报告 length 与 latency，并区分两个 reduction percentages。
3. 报告 token totals 和明确标注为 illustrative 的 cost estimate。
4. 中性报告 rating SD 和 Levene result。
5. 用一段简短 diagnostic 说明旧 all-seven anomaly 已消失、简单 content-homogeneity hypothesis 未获支持，且当前设计无法识别剩余 compression 的成因；不加入额外主表。
6. 以 operational conclusion 结束，而不是推断 reasoning quality。

## 4.4.8 RQ4 回答

> Schema-constrained reviewing substantially reduced output length, observed latency and token use, while producing a narrower rating distribution whose normative meaning cannot be determined from variance alone.

### 避免的表述

- “Structured saved 52.7% on every efficiency metric.”
- “Lower variance proves greater consistency.”
- “Lower variance proves cognitive damage.”
- “The experiment measured compute time directly.”
- “The estimated token price is the actual model cost.”

---

# 4.5 Cross-RQ synthesis

## 本章末尾 summary table

| RQ | Main result | 关于 structure 的回答 | Evidential boundary |
|---|---|---|---|
| RQ1 | 两种 setup 均发生 compliance；Structured 的 rating 与 MF effects 较小 | Partial attenuation, not defence | Setup-specific measurement procedures |
| RQ2 | Logic 与 Format responses 没有可靠差异 | No improvement in defect discriminability | Output sensitivity, not internal awareness |
| RQ3 | Free 的 operational total aspect count 更高 | Different coverage profile | Manual validation pending; count ≠ quality |
| RQ4 | Structured 更短、更快、token 更少，rating distribution 更窄 | Strong operational efficiency advantage | Dispersion has ambiguous meaning |

## Synthesis 段落规划

1. 回到 principal goal：structure 是否缓解已知缺陷。
2. 说明 structure attenuated selected injection effects，但没有形成 robust defence。
3. 说明 structure 没有改善 logic-specific discriminability。
4. 将 RQ3/RQ4 定位为 secondary properties：改变 coverage profile，并显著提高 operational efficiency。
5. 过渡到 Chapter 5：output organisation 不能替代 input security 或 scientific-reasoning safeguards。

### Chapter-level conclusion

> Structure changed the magnitude, distribution and cost of generated reviews, but structure alone did not resolve the central robustness and scientific-reasoning failures investigated in this study.

---

# Figures and tables 制作清单

## 已有 figures

- [x] `rq1_slope_graph.png`
- [x] `rq1_bubble_compliance.png`
- [x] `rq1_scatter_integrity.png`
- [x] `rq2_discriminability_box.png`
- [x] `rq3_comprehensiveness_stacked.png`
- [x] `rq4_tradeoffs.png`

## Final dissertation figures 所需修改

- [ ] RQ2：替换或移除 one-sample stars，突出 Logic-versus-Format tests。
- [ ] RQ3：将 title/caption 从 comprehensiveness 改为 aspect coverage。
- [ ] RQ4：将 cognitive variance/cost 改为 rating dispersion。
- [ ] 所有 captions 标注 measurement source：Free Judge versus Structured native。
- [ ] 所有 statistical annotations 与更新后的 `outputs/stats/` CSV 保持一致。

## 正文阶段需要制作的 tables

- [ ] RQ1 paired means 与 attenuation contrasts。
- [ ] RQ2 Logic-versus-Format contrasts，包括 CI、Holm p 和 `d_z`。
- [ ] RQ3 Original-condition coverage profile。
- [ ] RQ4 efficiency/token/dispersion summary。
- [ ] Cross-RQ summary table。
- [ ] Manual-validation agreement table。

---

# 写作状态清单

- [x] RQ1 descriptive statistics available。
- [x] RQ1 direct attenuation tests available。
- [x] RQ2 paper-level paired tests available。
- [x] RQ2 CI、effect size 和 Holm correction available。
- [x] RQ3 Original-condition descriptive counts available。
- [x] RQ4 word、latency、token 和 rating-distribution statistics available。
- [x] RQ4 rating-frequency、within-paper sensitivity 和 lexical-overlap diagnostics available。
- [x] Auxiliary manual-validation coding completed and inserted；LLM assistance 与 author review 必须在 Methods/Limitations 中披露。
- [x] RQ2 target-defect audit 与 RQ3 count-granularity case inspection completed；记录见 `outputs/manual_validation/qualitative_case_audit.md`。
- [ ] Final figures regenerated with corrected annotations and terminology。
- [ ] Dissertation prose drafted from this detailed outline。
