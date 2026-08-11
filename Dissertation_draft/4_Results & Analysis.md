# 第4章细纲：Results and Analysis

> **文件性质：** 本文件是 Chapter 4 的详细细纲和写作蓝图，不是 dissertation 正文。
>
> **当前分析口径：** 全章严格遵循已完成并经导师确认的实验流程。Free metrics 由独立 Judge 提取；Structured metrics 使用 native Pydantic self-report。Auxiliary condition-masked common coding 为两种 outputs 提供共同 measurement rubric，用于校准并部分弥合 setup-specific measurement gap，同时保留 primary pipelines 的实际使用逻辑。
>
> **章节职责：** Chapter 4 承担全文的主要 empirical argument。每个 RQ 都在本章完成 quantitative finding、measurement triangulation、qualitative/case mechanism analysis 与 RQ conclusion 的闭环。Chapter 5 不重新分析这些数据，而是在这些结论之上讨论与 prior work 的关系、总体意义、scope boundaries 和 future work。
>
> **叙事原则：** 结果围绕本文成立的贡献组织：Structured reviewing 对部分 injection effects 提供了显著 attenuation，带来明确的 efficiency/concision gains，并在 common coding 中保留相当的 recorded coverage；与此同时，RQ2 精确定位了仅靠 output structure 不能解决的 target-defect detection problem。边界说明只在影响 claim validity 时出现，不把章节写成自我审查清单。

---

## 4.0 本章目的与叙事顺序

### 本章目的

本章首先检验 schema-constrained reviewing 能否缓解 LLM peer review 的两类已知缺陷，再说明这种 setup 带来的 coverage profile 与 operational trade-offs。核心故事不是简单判断 Structured “更好或更差”，而是识别 structure 在哪些维度产生了可重复的价值、哪些风险需要互补机制处理。

### 证据层级

1. **Primary：RQ1 — Manipulation robustness。**
2. **Primary：RQ2 — Logic-defect discriminability。**
3. **Secondary：RQ3 — Review-aspect coverage。**
4. **Secondary：RQ4 — Operational efficiency and rating dispersion。**
5. **Measurement triangulation：** 5-paper condition-masked common coding of both Free and Structured outputs。

### 开头统一说明 measurement convention

- Free-form reviews：由 `deepseek-v4-flash` Judge 提取 rating 和 aspect counts。
- Structured reviews：由 native Pydantic fields 提供 rating 和 aspect counts。
- Injection-compliance scores：两种 setup 均由 Judge 提取。
- 因此，两种 setup 的 rating/aspect counts 使用不同的 operational measurement procedures。
- 该差异不妨碍各 setup 内部的 paired comparison；跨 setup effect magnitude 属于完整 pipeline-level comparison。
- Common auxiliary coding 将同一 rubric 应用于两类 review texts，为主要指标提供 convergent evidence，并识别 weakness count 这类 measurement-sensitive dimensions。

### 统一论证顺序

1. 报告 primary quantitative pattern 与直接 comparison。
2. 使用 common coding / human benchmark 检查该 pattern 是否依赖 measurement procedure。
3. 使用 targeted case 或 qualitative inspection 解释 aggregate pattern 在 review text 中如何体现，并排除最关键的替代解释。
4. 综合上述证据直接回答 RQ，明确 structure 的价值和适用范围。

### 建议开头段落结构

1. 重申所有核心比较均为 within-paper comparison。
2. 说明 N = 30，并区分 Injection track 与 Counterfactual track。
3. 说明 Results 按 RQ 而非 pipeline step 组织。
4. 说明描述性图表同时配合直接回答 RQ 的 effect estimate、95% confidence interval 和 multiplicity-adjusted p-value；不堆叠重复统计量。

> 本章可以解释数据直接支持的 output-level mechanism，例如 target omission、critique–rating decoupling、segmentation sensitivity 和 criticism retention。只有无法由当前 outputs 验证的认知原因、与 prior work 的关系、system-design implications 和 future directions 才留到 Chapter 5。

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

| Contrast | Mean difference | 95% CI | Holm p |
|---|---:|---:|---:|
| Free Δ − Structured Δ | 0.93 | [0.61, 1.26] | < .001 |

### Evidence-based analysis

**Table 4.1 — RQ1 Attack Effects and Attenuation (N = 30 paired).**

| Metric | Free Δ [95% CI] | Struct Δ [95% CI] | Contrast (Free−Struct) [95% CI] | Holm p |
|---|---:|---:|---:|---:|
| Rating (1–10) | +1.90 [+1.58, +2.22] | +0.97 [+0.76, +1.17] | +0.93 [+0.61, +1.26] | < .001 |
| Methodological Flaws | −3.20 [−3.92, −2.48] | −1.83 [−2.32, −1.34] | −1.37 [−2.29, −0.44] | .011 |
| Strengths | +1.10 [−1.06, +3.26] | +2.10 [+1.68, +2.52] | −1.00 [−3.22, +1.22] | .364 |
| Weaknesses | −2.70 [−3.43, −1.97] | −1.20 [−1.68, −0.72] | −1.50 [−2.37, −0.63] | .004 |
| Injection Compliance (0–10) | +5.30 [+4.33, +6.27] | +2.40 [+1.43, +3.37] | +2.90 [+1.69, +4.11] | < .001 |

*Note.* Free metrics extracted by independent Judge (`deepseek-v4-flash`); Structured metrics from native Pydantic self-report. Original means: Free Rating = 6.23, Struct Rating = 5.90; Free MF = 4.27, Struct MF = 8.07. Contrasts tested with paired *t*-test; Holm correction applied across 5 contrasts.

- Prompt injection 显著提高了两种 setup 的 ratings。
- Structured 条件下的 rating increase 显著小于 Free。
- 因此，Structured **attenuated rating inflation but did not prevent it**。

### 对应图

`outputs/figures/rq1_slope_graph.png`

### 正文段落规划

1. 介绍 paired rating comparison 与 slope graph。
2. 报告 Original/Injected means 和两种 setup 内部的 ATE。
3. 报告 Free-versus-Structured delta comparison、95% CI 和 Holm-adjusted p-value。
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

### Evidence-based analysis

- Payload 不仅改变 rating，也改变了 praise 与 criticism 的相对构成。
- Free 呈现更大的 weakness reduction 和 compliance-score increase。
- 由于 Structured 的 strengths increase 更大，不能笼统声称 Structured 在所有方面都更少服从攻击。
- Weakness reduction 在 automatic measure 中很明显，但 auxiliary common coding 未能可靠复现其幅度；因此它只作为后续 RQ3 measurement-sensitivity 分析的伏笔，不单独支撑 RQ1 结论。

### 正文段落规划

1. 说明 bubble chart 的 axes 和 bubble size 如何对应 payload 的三个目标。
2. 分别报告 weakness 与 strength changes，不要合并成单一 compliance 结论。
3. 使用 Judge-derived compliance score 作为 convergent evidence。
4. 简短指出 weakness count 的 common-coding agreement 较差，不在 RQ1 中延伸解释，留待 RQ3 处理。
5. 明确指出不同 metrics 上的 attenuation 具有 heterogeneity。

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

| Contrast | Mean difference | 95% CI | Holm p |
|---|---:|---:|---:|
| Free Δ − Structured Δ | -1.37 | [-2.29, -0.44] | .005 |

### 呈现方式

- 本结果并入 RQ1 主统计表，不在正文单独使用 `rq1_scatter_integrity.png`。
- 该 scatter plot 的 trend line 检验的是 score change 与 MF-count change 的相关趋势，并不直接呈现本节的 paired reduction 与 Free–Structured attenuation contrast；单独保留会增加视觉负担。
- 文件可留作 internal diagnostic，但正文不使用 “MF blindness” 作为图题或结论标签。

### Evidence-based analysis

- Injection 显著减少了两种 setup 中报告的 methodological flaws。
- Structured 条件下的 reduction 显著小于 Free。
- 正文使用 “reduced methodological criticism” 或 “methodological-flaw reporting decreased”。
- 不使用 “MF blindness”；正文只报告 methodological-flaw reporting decreased。

### 正文段落规划

1. 报告两种 setup 的 paired reductions 和方向性数量。
2. 报告 attenuation contrast、95% CI 和 Holm-adjusted p-value。
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

Measurement agreement 的关键边界为：Free rating MAE = 0.64、`ρ = .83`；Free/Structured compliance MAE 分别为 1.90/1.50；Free/Structured weakness MAE 分别为 6.28/3.40，且 rank correlations 接近 0 或为负。由于 RQ1 的 Injection track 与 RQ3 的 Original counterfactual track 不同，正文在此只报告 weakness count 未可靠复现，不进行跨 track 强解释。

### 定量结果留下的解释缺口

RQ1 的 paired tests 已确定 Structured 条件下部分 attack effects 较小；接下来的 matched case 将这一 aggregate result 连接到 review text，检验 attenuation 是否表现为 mandatory criticism fields 中保留更多 substantive concerns：

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

> Schema-constrained reviewing significantly attenuated rating inflation and the reduction in methodological-flaw reporting, while also producing a smaller injection-compliance shift. Both setups remained behaviourally susceptible, so structure provided measurable partial mitigation rather than complete defence.

### Claim boundary

结论聚焦完整 Structured setup 对部分 attack effects 的显著 attenuation；不把该结果扩大为 complete defence，也不将 effect 唯一归因于 JSON serialization。

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

| Setup | Logic mean Δ | Format mean Δ | Logic − Format | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|
| Free | +0.66 | +0.77 | -0.11 | [-0.58, 0.36] | 1.000 |
| Structured | -0.43 | -0.63 | +0.20 | [-0.08, 0.48] | .647 |

### 解释方向

- Free 在 Logic perturbations 后平均提到更多 methodological flaws，但 Format perturbations 后的增加略大。
- Structured 在两种 perturbation classes 后都报告了更少的 methodological flaws。
- Holm correction 后，两种 Logic–Format contrasts 均不显著。
- 两个 Logic–Format contrasts 接近 0，且 confidence intervals 均跨 0。
- Methodological-flaw count 上升不能称为“惩罚”，因为 rating 并未下降。

## 4.2.3 Rating response

### 统计结果

**Table 4.2 — RQ2 Discriminability: Logic vs. Format Perturbations (N = 30, paper-level aggregates).**

| Metric | Setup | Logic mean Δ | Format mean Δ | Logic−Format contrast [95% CI] | Holm p |
|---|---:|---:|---:|---:|---:|
| Rating (1–10) | Free | +0.02 | +0.03 | −0.00 [−0.22, +0.21] | 1.000 |
| Rating (1–10) | Structured | +0.01 | +0.03 | −0.01 [−0.11, +0.08] | 1.000 |
| MF count | Free | +0.66 | +0.77 | −0.11 [−0.58, +0.36] | 1.000 |
| MF count | Structured | −0.43 | −0.63 | +0.20 [−0.08, +0.48] | .647 |

*Note.* Logic and Format columns report mean changes from each paper's Original review, averaged within perturbation class before the paper-level comparison. Free metrics: Judge-extracted; Structured metrics: Pydantic self-report. All pairwise tests use two-tailed paired *t*-tests. Holm correction was applied across the four Logic–Format contrasts. No contrast approaches significance at α = .05.

### 解释方向

- 两种 setup 的 ratings 对两类 perturbations 都几乎没有变化。
- 两种 setup 都没有对 logic-critical edits 施加 evaluative consequence。
- 接近 0 的 raw-scale contrasts 与跨 0 的 confidence intervals 表明结果不只是“未达到显著”，而是没有观察到有意义的方向性差异。

## 4.2.4 对应图及其限制

`outputs/figures/rq2_discriminability_box.png`

当前图显示 direct Logic–Format comparisons，不再使用各 box 相对 0 的 one-sample stars；四项 annotation 已统一为 Holm-adjusted p-values（Free MF = 1.000、Structured MF = .647、Free rating = 1.000、Structured rating = 1.000），与 Table 4.2 一致。

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

因此，就 RQ2 的核心问题而言，首要发现是 **target-defect detection 不可靠**：至少一半 reviews 未识别被植入的逻辑错误。Critique–rating decoupling 是次级机制：它解释了为什么部分已经识别缺陷的 reviews 仍然没有下调 primary rating。

### 正文代表性案例

使用 Case 3（`2024.emnlp-main.1123`, `blueprint_result`）：修改后的段落宣称 domain-specific models underperform，但同段仍保留 `xlmt_large surpasses xlmr_large` 的正向结果。Free 与 Structured reviews 均明确指出该 claim–result contradiction，但两种 setup 的 primary ratings 相对 Original 均未改变。该例直接说明 critique 可以出现而不产生 score consequence。

Case 5 不得称为 hallucination：counterfactual paper 中仍有未修改的 3.2；真正遗漏的是新增 2.0 与既有 3.2 之间的 internal inconsistency。

## 4.2.6 正文段落规划

1. 定义 paper-level aggregation，并解释为什么独立分析单位仍为 N = 30 papers。
2. 报告 Free/Structured 的 methodological-flaw contrasts。
3. 报告 rating contrasts。
4. 强调 Holm correction 后四项比较均不显著，raw-scale contrasts 接近 0，且 confidence intervals 均跨 0。
5. 用 5-case audit 明确主次：target-defect omission 是首要失败，critique–rating decoupling 是已识别 cases 中的辅助解释。
6. 加入 Case 3，区分 target detection、critique expression 与 primary score consequence。

## 4.2.7 RQ2 回答

### 可用于正文的小结句

> Neither free-form nor schema-constrained reviewing demonstrated reliable logic-specific discriminability, and schema constraint did not improve the response to logic-critical edits relative to surface perturbations.

补充机制句：

> The post-unblinding audit identified unreliable target-defect detection as the primary failure; critique–rating decoupling provided a secondary explanation for the detected cases that still received no rating penalty.

### Claim boundary

结论比较两种 setup 的 output-level Logic–Format response，并由 target audit 解释 observable detection/score behaviour；不推断模型不可观测的 internal awareness。

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

**Table 4.3 — RQ3 Evidence Triangulation: Aspect Coverage across Three Measurement Layers.**

| Panel | Source and measurement | Analysis unit | Strengths | Weaknesses | MF | Total |
|---|---|---:|---:|---:|---:|---:|
| A: Operational | Free, Judge-extracted | 30 Original papers | 6.5 | 11.1 | 8.6 | 26.2 |
| | Structured, Pydantic-native | 30 Original papers | 5.9 | 7.6 | 8.8 | 22.3 |
| B: Common coding | Free, dissertation-author coding | 5 Original papers | 4.4 | 3.4 | 10.6 | 18.4 |
| | Structured, dissertation-author coding | 5 Original papers | 6.6 | 2.2 | 10.4 | 19.2 |
| C: Human benchmark | Human, dissertation-author coding | 11 papers / 45 reviews | 3.02 | 1.93 | 2.79 | 7.73 |
| | Structured, Pydantic-native | 11 matched Original papers | 5.91 | 7.45 | 9.45 | 22.82 |
| | Free, Judge-extracted | 11 matched Original papers | 5.91 | 11.09 | 8.45 | 25.45 |

*Note.* Panel A reports the two operational pipelines. Panel B applies one common rubric to both setups in the same five-paper Original-only auxiliary sample. Panel C is a matched external benchmark: multiple Human reviews are first averaged within paper, so all three rows use 11 papers as the comparison unit. Human aspects were coded by the dissertation author. Measurement procedures differ between Panels A/C sources; cross-source magnitudes are descriptive reference points rather than directly equivalent measurements.

### 主要模式

- Free 的 extracted total aspect count 更高。
- 最大差异来自 weakness count。
- Original condition 下，两种 setup 的 methodological-flaw counts 接近。
- Structured 保证了 machine-readable fields，但没有产生更高的 total recorded aspect count。

## 4.3.3 对应图

`outputs/figures/rq3_aspect_coverage_stacked.png`

### 必须修正的命名

- 正文统一称为 **review-aspect coverage**。
- 当前 title 已使用 aspect coverage；caption 仍需说明 Free 为 Judge-extracted、Structured 为 native counts，并明确 count 不等于 correctness 或 non-redundant comprehensiveness。

## 4.3.4 Measurement triangulation and comparability

- Primary counts 反映两条实际 operational pipelines：Free counts 由 Judge 从 prose 中提取，Structured counts 是 native list lengths。
- 为校准两种 measurement sources，5-paper auxiliary sample 对 Free 和 Structured review texts 使用同一 coding rubric。该 common-measure layer 部分弥合了 measurement gap，使得 RQ3 不只依赖 setup-specific automatic counts。
- 因此 RQ3 同时报告 operational comparison 与 common-coded comparison：前者展示完整 pipeline 的输出 profile，后者检查该 profile 在统一尺度下是否保持。
- Item count 仍不等同于 correctness、importance 或 non-redundancy；这些更高层的 quality questions 由 qualitative inspection、common coding 与 human-reference benchmark 联合检验。

## 4.3.5 Common-measure triangulation result

Table 4.3 Panel B 报告 Original-only 5-paper matched sample；对应的 operational automatic totals 为 Free 29.2、Structured 24.0。

与 automatic analysis 中 Free total 较高不同，共同 coding 下两个 setup 的 total aspect count 基本相同，Structured 仅高 0.8。共同 coding 仍显示不同 profile：Structured strengths 较多、Free weaknesses 略多、MF 几乎相同。

Agreement diagnostics 显示：

- Free strengths：MAE = 1.88，`ρ = .58`；Structured strengths：MAE = 0.72，`ρ = .80`。
- Free MF：MAE = 2.40，`ρ = .69`；Structured MF：MAE = 0.56，`ρ = .87`。
- Free weaknesses：MAE = 6.28，`ρ = -.05`；Structured weaknesses：MAE = 3.40，`ρ = -.18`。

这一 common-measure result 补全了 automatic comparison 的解释：Free > Structured 的 total ordering 不是跨 measurement procedures 都稳定的现象，差异主要集中在 measurement-sensitive 的 weakness dimension。同时，统一 coding 下 Structured 并未呈现较低的 total recorded coverage，这为“较短 Structured reviews 未明显牺牲 aspect coverage”提供了有限但直接的证据。RQ3 因而应解释为 **measurement-sensitive aspect profiles with comparable common-coded totals**，而不是 Free 的稳定 total-coverage advantage。

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

## 4.3.7 Human-review aspect benchmark

### 目的

引入真实 human peer reviews 作为 external reference，检查正常 peer-review practice 下的 aspect profile，并将其与相同 papers 的 Free/Structured operational profiles 作描述性对照。该 benchmark 用于判断模型输出的 aspect counts 是否处于与 human reviewing 相近的数量级，并辅助解释 Free 的较高 automatic count；它不替代 5-paper common-measure comparison。

### 样本与统计口径

- 在 30 篇实验 papers 中，11 篇具有符合既定标准的公开 primary human reviews；不使用其他 papers 替换无公开 review 的 papers。
- 共纳入 45 份 human reviews；所有 strengths、weaknesses 和 methodological flaws 均由 dissertation author 人工编码，不由 LLM Judge 编码。
- 同一 paper 的多份 human reviews 先取 paper-level mean，再跨 11 papers 汇总，使比较单位与每篇 paper 各一份 Free/Structured Original review 一致。
- Human counts 与 Free/Structured operational counts 的 measurement procedures 不同，因此三组比较只作为 external descriptive benchmark；关于 Free-versus-Structured coverage 的主要校准证据仍是 5-paper common coding。

### 描述性结果（11 篇 matched Original papers）

三组 paper-level aspect profiles 已统一收入 Table 4.3 Panel C，不再另设重复表格。

### 解释

- Human reviews 在 dissertation-author rubric 下平均记录约 7.7 个 aspects，明显少于两种 LLM operational profiles；因此，不能将模型生成的大量 item counts 直接等同于正常 human-review coverage。
- Free 与 Structured 的 matched operational total difference 仍较小于两者与 Human 的距离，并且 Free 的额外 count 主要来自 weaknesses（11.09 vs 7.45）；strengths 相同，Structured 的 MF count 反而略高。
- 该 profile 与 RQ3 的主结果一致：Free 的 apparent total-count advantage 集中在 measurement-sensitive weakness dimension，而不是三个 dimensions 一致增加。
- 由于 Human、Free 与 Structured 并非由同一 measurement procedure 编码，本表不能单独证明 Free 的额外 items 是 repetition，也不能证明 Structured 具有 human-equivalent substantive coverage。
- 将该 external benchmark 与 5-paper common coding 的 comparable totals、Pair A 的 segmentation sensitivity 以及 RQ4 的 length benchmark 联合起来，可作有限解释：Free 的额外长度和 count **are consistent with unnecessary granularity or redundancy**, 而不是稳定证据表明其具有更全面的 substantive coverage。

## 4.3.8 正文段落规划

1. 说明为什么只使用 Original condition。
2. 报告三个 aspect dimensions 和 total counts。
3. 指出 automatic total difference 主要由 extracted weaknesses 驱动，但 common coding 未复现 total ordering。
4. 报告 Pair A 的 full totals 与 critical-aspect counts，明确 category reassignment 和 segmentation sensitivity。
5. 说明 count-only Judge 无法提供 item-level over-segmentation proof。
6. 加入 11-paper Human external aspect benchmark，并在表中明确三组 measurement source。
7. 联合 common coding、Pair A 与 Human benchmark 得出有限的 redundancy/granularity interpretation，同时明确该解释不是 item-level causal proof。

## 4.3.9 RQ3 回答

### 可用于正文的小结句

> Under the setup-specific operational measures, free-form reviews yielded a higher total aspect count, driven primarily by extracted weaknesses. Common-measure auxiliary coding partially bridged the measurement gap and showed comparable totals across setups, indicating that schema-constrained reviews did not exhibit a clear loss of recorded aspect coverage in the matched sample and that the automatic ordering was concentrated in a measurement-sensitive dimension.

补充解释句：

> Human reviews contained substantially fewer author-coded aspects than either operational LLM profile. Together with the common-coded comparison and qualitative inspection, this pattern is consistent with the Free setup's additional counts reflecting greater segmentation or redundancy, rather than a stable advantage in substantive coverage.

### Claim boundary

RQ3 的贡献是区分 operational count profile 与 common-coded coverage：Structured 在 audited sample 中没有显示 clear recorded-coverage loss，而 Free 的额外 counts 集中于 measurement-sensitive weaknesses。该结果不被扩展为一般 review-quality ranking。

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

**Table 4.4 — RQ4 Trade-offs: Efficiency and Rating Dispersion (N = 240 across all conditions).**

*Panel A: Operational Efficiency.*

| Metric | Free | Structured | Ratio (Struct/Free) |
|---|---:|---:|---:|
| Words (μ) | 1884 | 793 | 42% |
| Words (median) | 1872 | 782 | — |
| Words (SD) | 183 | 92 | — |
| Latency (μ) | 29.1 s | 13.7 s | 47% |
| Latency (median) | 28.9 s | 13.6 s | — |
| Output tokens (μ per review) | 2354 | 1113 | 47% |
| Output tokens (total) | 565,075 | 267,167 | 47% |
| Rating SD (all 240 ratings) | 1.27 | 0.93 | — |
| Rating per-paper σ (μ) | 0.72 | 0.35 | — |
| Rating per-paper σ (median) | 0.67 | 0.35 | — |

*Panel B: 11-Paper Matched Human Length Benchmark.*

| Source | Analysis unit | μ words | Median | Range | vs. Human |
|---|---:|---:|---:|---:|---:|
| Human (OpenReview) | 11 papers (45 reviews) | 387.5 | 345.5 | [250, 603.5] | 1.00× |
| Structured (Original) | 11 papers | 851.4 | 812 | [754, 1173] | 2.20× |
| Free (Original) | 11 papers | 1874.4 | 1855 | [1679, 2051] | 4.84× |

*Note.* Panel A uses all 240 rows (30 papers × 8 conditions). Latency is end-to-end API wall-clock time. Global rating variance is compared using Levene's test; mean and median paper-level SDs and invariant-paper counts provide a matched descriptive view of within-paper dispersion. Panel B uses 11 ICLR 2025 / NeurIPS 2024 papers with publicly available official reviews and Original-condition LLM outputs. Multiple Human reviews are averaged within paper before the three-source comparison. Human review texts exclude author rebuttals, section headers, rating labels and metadata.

### 解释方向

- 在本实验 API conditions 下，Structured reviews 显著更短、更快。
- Length、latency 与 token use 呈一致的 operational reduction；exact summaries 由 Table 4.4 报告，无需额外 correlation figure。
- 正文称其为 observed end-to-end API latency，而不是 pure compute time。
- 不将结果扩展为对未知 backend efficiency 的因果结论。

### 4.4.2A Human-review length benchmark

#### 动机与分析问题

导师建议引入一组真实 human peer reviews 作为参照，检查 Free 与 Structured reviews 的长度差异是否可能反映 verbosity / information redundancy，而不只是 operational efficiency。核心问题为：**human-review length distribution 更接近 Free 还是 Structured？**

#### 参照样本与可比性

**样本收集标准与最终范围：**

1. 目标对象固定为实验中的同一批 30 篇 Original papers，不用其他 papers 替换无公开 review 的 paper。
2. 只从可核验的 official/public review page 收集 primary reviewer reports，并保存 `paper_id`、`review_id` 和 `source_url`。
3. 使用 rebuttal/author response 前的最初完整 review version；排除 meta-review、area-chair summary、decision、author response、public comments、ethics-only review 和纯 rating 记录。
4. 同一 paper 的所有 eligible primary reviews 均纳入，不按长度、评分或评价倾向挑选；若无公开 review，记为 missing。
5. 长度只计算 textual evaluation body；对分栏 review，按固定顺序合并 summary、strengths、weaknesses/questions 等文本字段，不计入 rating、confidence、栏目标题和页面 metadata。
6. 每份 review 用与 Free/Structured 相同的 word-count procedure 计算 length；同一 paper 有多份 reviews 时取 paper-level mean，使 Human、Free 和 Structured 均以 paper 为比较单位。
7. 最终在 30 篇实验 papers 中找到 11 篇具有 eligible public reviews，共收集 45 份 human reviews；benchmark 只比较这 11 篇 matched papers 的 `Original` Human、Free 与 Structured reviews，不与全部 240 条 perturbed outputs 混合。

分析真正需要的 outcome 只有 `word_count`，以及 RQ3 的三个 aspect counts。`paper_id/review_id/source_url/review_text` 只是用于可追溯性的必要记录，不是额外分析变量。

#### 最终比较口径

- **RQ4 primary efficiency result：** 保留全部 240 rows，回答完整 experimental pipeline 中的 output-length、latency 与 token-use differences；对应均值仍为 Free 1883.7 words、Structured 793.3 words。
- **Human benchmark：** 使用 11 篇 matched Original papers，并以 paper-level mean 为分析单位；这是 Human/Free/Structured length comparison 的主口径。
- 45 份 human reviews 的 review-level statistics 用于描述 benchmark sample，但不把同一 paper 内多份 reviews 当作 45 个相互独立的 matched observations。

#### 描述性结果

Human sample: N = 45 reviews across 11 papers, μ = 388 words, median = 376, SD = 167, range = [146, 901]. See Table 4.4 Panel B for matched paper-level comparison with Free and Structured.

At the matched paper level, Human reviews averaged 387.5 words, compared with 851.4 for Structured and 1874.4 for Free. Structured was therefore directionally closer to Human reviewing practice, although it remained approximately 2.2 times as long. Length alone does not establish equal information content; combined with the common-coded RQ3 totals and the qualitative granularity check, the pattern is consistent with part of Free's additional length reflecting unnecessary verbosity or finer segmentation rather than a stable substantive-coverage advantage.

所有 11 篇 matched papers 中，Human paper-level mean 均短于 Structured，而 Structured 均短于 Free。Structured 相对 Free 缩短约 54.6%，但仍约为 Human 的 2.2 倍；Free 接近 Human 的 4.8 倍。

#### 最小图表方案

- `rq4_tradeoffs.png` 左 panel 已采用 **matched paper-level grouped bar chart**：每篇 paper 并列显示 Human、Structured 与 Free word count，并叠加 group mean reference lines。
- 移除现有 words–latency scatter panel；length 与 latency 的 exact statistics 放入 RQ4 compact table，相关关系在正文一句报告即可。
- 不另建大型 human-length table；将 N、mean、median 和 range 作为 RQ4 compact table 的一个 panel。
- RQ3 只保留一张 compact triangulation table；无需为 Human aspects 另作图，也不做额外复杂显著性检验。

#### 结果解释

1. Human reviews 并非仅略短于 Free，而是比两种 LLM outputs 都明显更短；因此，Structured 应描述为 **substantially more concise than Free and directionally closer to human reviewing practice**，不能称为 human-like in absolute length。
2. Free 的近五倍 human length 表明其 verbosity 远超该 external benchmark。结合 RQ3 中 common-coded totals 相当、Free 的 operational count difference 主要由 weaknesses 驱动，以及 Pair A 的 segmentation sensitivity，可将 Free 的额外长度解释为 `consistent with unnecessary verbosity or granularity`。
3. Human benchmark 本身不能证明 Free 多出的所有文字都是冗余，也不能证明 Structured 在更短篇幅中保留完全相同的信息；支持有限 redundancy interpretation 的是多种证据的联合，而不是 length alone。

#### 证据边界

- Human length 与 RQ3 common-coded aspects 联合解释：Human 在长度上远接近 Structured，而 5-paper common coding 未显示 Structured total coverage 明显较低，支持 Free possible unnecessary verbosity 的有限解释。
- Human aspect 与 length 可以判断 Structured 的 brevity 是否伴随明显 coverage loss，但**不能单独区分较窄 rating dispersion 究竟代表 consistency 还是 reduced sensitivity**。该判断需要 human ratings 或外部 review-quality / sensitivity criterion。

## 4.4.3 Token use 与 estimated cost

Token totals 与 per-review means 已收入 Table 4.4 Panel A。Structured output tokens 从 565,075 降至 267,167，平均每份 review 从 2,354 降至 1,113，减少 52.7%。若正文需要帮助读者理解量级，可用 `$10/1M output tokens` 明确标注为假设口径，给出 illustrative estimate（Free $5.65；Structured $2.67），不另设表格。

### 报告边界

- 明确 tokenizer 为 `o200k_base`。
- 将金额标记为基于 `$10/1M output tokens` 假设的 illustrative estimate。
- 除非有真实 invoice 或正式 pricing 支持，不将其描述为 `gpt-5.4` 的 verified billing cost。

## 4.4.4 Rating dispersion

Table 4.4 Panel A 报告 global 与 within-paper dispersion。Free / Judge-extracted ratings 的 mean = 5.35、median = 6、SD = 1.27；Structured / native ratings 的 mean = 5.72、median = 6、SD = 0.93。

- Levene test：`p < .001`。
- Structured ratings 呈现更窄的 distribution。
- 这是 distributional difference，不是 cognitive variance 的直接证据。
- Lower dispersion 可能表示 desirable consistency、scale compression 或 reduced sensitivity；当前设计无法区分这些解释。
- 两组 rating 的 extraction procedures 不同，因此不进行强 normative comparison。
- Human benchmark 显示 Structured 在长度上明显更接近 Human；结合 5-paper common coding 未发现其 total recorded coverage 明显下降，可将 Structured 解释为相对更 concise。但这不会自动改变 rating-dispersion 的中性结论。

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

### Claim boundary

Diagnostic 排除了“reviews 近乎复制”这一简单解释，并确认 Structured ratings 更集中；其 normative meaning 不承担 RQ4 efficiency/concision contribution 的证明责任。

## 4.4.6 对应图

`outputs/figures/rq4_tradeoffs.png`

### 最终 panel 设计

- 保留右侧 **Rating Dispersion** panel；caption 说明 Free ratings 为 Judge-extracted、Structured ratings 为 native。
- 左侧已替换为 11-paper matched Human/Structured/Free word-count grouped bar panel；Length–latency relationship 用 RQ4 主表和正文报告即可，无需同时占用主图。
- 这样 human benchmark 不新增独立 figure，RQ4 一张双 panel figure 同时表达 relative concision 与 rating-dispersion trade-off。

## 4.4.7 正文段落规划

1. 介绍基于全部 240 rows 的 engineering comparison。
2. 报告 length 与 latency，并区分两个 reduction percentages。
3. 引入 11-paper matched human-review length benchmark，报告 Human 387.5、Structured 851.4、Free 1874.4 words，并说明 Structured 虽更接近 Human，但仍约为 Human 的 2.2 倍。
4. 将 human-length result 与 RQ3 substantive coverage / non-redundancy evidence 联合解释；不根据长度单独宣称 Free 冗余。
5. 报告 token totals 和明确标注为 illustrative 的 cost estimate。
6. 中性报告 rating SD 和 Levene result。
7. 用一段简短 diagnostic 说明旧 all-seven anomaly 已消失、简单 content-homogeneity hypothesis 未获支持，且当前设计无法识别剩余 compression 的成因；不加入额外主表。
8. 以 operational conclusion 结束，而不是推断 reasoning quality。

## 4.4.8 RQ4 回答

> Schema-constrained reviewing substantially reduced output length, observed latency and token use. In the matched human benchmark, Structured reviews were markedly closer to human-review length than Free reviews, although they remained approximately twice as long. Together with comparable common-coded aspect totals, this supports greater concision without a clear loss of recorded coverage in the audited sample; however, the normative meaning of the narrower rating distribution cannot be determined from variance or length alone.

### Claim boundary

RQ4 的强结论是 observed output length、latency、token use 和 human-relative concision；rating dispersion 作为伴随的 distributional property 单独报告，illustrative cost 明确标注估算口径。

---

# 4.5 Cross-RQ synthesis

## 呈现方式

本节不再制作 cross-RQ summary table，因为四个 RQ 的 exact results 已分别由 Figures 4.1–4.5 与 Tables 4.1–4.4 报告。结尾使用一个 synthesis paragraph 串联 primary findings 与 secondary trade-offs，避免逐项重复数字。

## Synthesis 段落规划

1. 回到 principal goal：structure 是否缓解已知缺陷。
2. 说明 structure attenuated selected injection effects，但没有形成 robust defence。
3. 说明 structure 没有改善 logic-specific discriminability。
4. 将 RQ3/RQ4 连成同一条 secondary analysis：Free 的 automatic aspect advantage 集中在 measurement-sensitive weaknesses，共同 coding 未显示 Structured total coverage 明显下降；与此同时，Human length benchmark 显示 Structured 虽仍长于 Human，但远短于 Free。三类证据联合支持 Structured 的 relative concision，并使 Free 的额外长度更符合 unnecessary verbosity/granularity，而不是稳定的 substantive-coverage advantage。
5. 过渡到 Chapter 5：output organisation 不能替代 input security 或 scientific-reasoning safeguards。

### Chapter-level conclusion

> Structure changed the magnitude, distribution and cost of generated reviews, but structure alone did not resolve the central robustness and scientific-reasoning failures investigated in this study.

---

# 剩余待办

## P0：正文写作

- [ ] 从本细纲起草 Chapter 4 正文，并在 Methods/Limitations 中披露 auxiliary coding 的 single-annotator design。
