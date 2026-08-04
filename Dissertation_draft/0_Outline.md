# Dissertation Master Outline

## Working title

### Recommended

**Structure Is Not a Safeguard: Efficiency, Rating Dispersion, and Robustness in LLM-Based Peer Review**

### Alternatives

- **Schema-Constrained but Still Vulnerable: Evaluating Structured LLM-Based Peer Review**
- **The Price of Structure: Efficiency and Robustness Trade-offs in LLM-Based Peer Review**
- **Structured but Not Secure: Testing Manipulation Robustness and Defect Discriminability in Automatic Peer Review**

> [修改] 删除标题中的 “LLM-as-a-Judge”。本项目的主要研究对象是作为 reviewer / automatic review generator (ARG) 的 LLM；Judge 只是对自由文本评审进行特征提取的测量工具。
>
> [修改] 将 “Cognitive Variance” 改为 “Rating Dispersion”。实验测量的是评分输出的离散程度，并没有直接测量模型的认知过程。

---

## Status markers used in this outline

- **[修改]**：原有内容或结论需要改写。
- **[新增]**：目前缺失、但为了建立完整论证必须加入。
- **[待分析]**：需要补做统计检验、稳健性分析或人工核验后才能定论。
- **[待补充]**：核心分析已经完成，但正文报告或辅助统计仍需补齐。
- **[计划]**：已经确定设计、尚未执行的辅助工作。
- **[已完成]**：分析已经执行，并存在可追溯的结果文件。
- **[限制]**：必须在正文中主动承认的证据边界。
- **[可保留]**：现有方向与证据基本一致，可据此展开正文。

---

# Central argument of the dissertation

现有研究已经分别证明：LLM 自动审稿系统容易受到隐藏文本注入操纵，也不能可靠地区分论文中的逻辑缺陷与意义不变的表面变化。然而，严格的 schema-constrained reviewing setup 是否会改变这两类失败，以及这种约束会带来哪些内容和工程层面的代价，仍缺少统一、受控的实证研究。

本研究在同一批论文、同一生成模型和严格的被试内设计下，对比 free-form 与 schema-constrained 两种审稿设置。结果应围绕以下中心论点组织：

> **Strict output structure substantially improves operational efficiency and changes the distribution of review content, but it does not constitute a reliable safeguard against prompt injection and does not improve the specific discrimination of logic defects over surface perturbations.**

必须避免把结果写成“结构化一定更差”或“JSON 严重损害推理能力”。当前实验更可靠地支持：

1. 两种审稿设置都容易受到白字 prompt injection；
2. Structured 条件下部分攻击效应较小，但攻击没有被阻止；
3. 两种设置都没有可靠地区分逻辑扰动和表面扰动；
4. Structured 输出显著更短、更快、token 消耗更低；
5. Structured 的评分分布更窄，但这究竟代表更高一致性还是更低敏感性，不能仅凭方差判断。

---

# 1. Introduction

## 1.1 Background and motivation

- 介绍 peer review 在科学质量控制中的作用，以及投稿数量增长、专家审稿资源不足和审稿成本上升的问题。
- 说明 LLM 已经被用于辅助或自动生成同行评审，形成 automatic review generators (ARGs)。
- 简要概括其潜在价值：规模化、速度、降低工作负担、快速反馈和较低边际成本。
- 强调同行评审是高风险判断任务：生成流畅、形式完整的 review 不等于能够进行可靠的科学评估。

> [新增] 背景不能只写“LLMs 快速发展”。需要先说明为什么学界有采用 ARGs 的现实动机，再说明为什么其失败会威胁 scientific integrity。

## 1.2 Two established risks

### 1.2.1 Manipulation vulnerability

- Ye et al. 证明隐藏在 PDF 文本层中的小号白字能够进入自动审稿管线所提取的文本，并显著操纵生成结果。
- 明确区分：
  - explicit manipulation：隐藏 prompt injection；
  - implicit manipulation：作者通过主动披露次要 limitations，引导 LLM 重复这些限制。
- 准确表述证据：显式注入可提高评分、减少批评并降低 LLM review 与 human review 的一致性；隐式操纵使 LLM 更倾向复述作者披露的 limitations。

> [修改] 不要直接声称隐式披露“导致 LLM 忽视核心问题”。原研究直接证明的是对作者披露内容的过度复述；忽视其他问题只能作为潜在风险或推论。

### 1.2.2 Failure to detect faulty research logic

- Dycke and Gurevych 将 paper soundness 表示为 method → result → conclusion → finding 的 research-logic structure。
- 其 counterfactual framework 对 soundness-critical 内容进行最小化编辑，并以 soundness-neutral 表面编辑作为控制。
- 可靠 ARG 应当对 logic-critical perturbations 产生比 surface perturbations 更强、更有方向性的反应。
- 现有结果显示，多种 ARG 的评分、sentiment 和 soundness-related aspects 都不能可靠地区分这两类扰动。

> [修改] 重点不是“模型有没有对逻辑扰动产生任何变化”，而是“对逻辑扰动的反应是否显著强于对意义不变的表面扰动”。这一定义直接决定 RQ2 的统计检验。

## 1.3 Research gap

建议采用以下逻辑，而不是笼统声称“此前没有研究 structured reviewing”：

1. Ye et al. 研究了 manipulation vulnerability，但没有把严格 schema-constrained output 作为核心对照条件。
2. Dycke and Gurevych 比较过 ZERO-GENERIC、ZERO-GUIDE 及其他 ARGs，因此不能声称 generic versus guided reviewing 从未被研究。
3. 现有研究没有在统一的被试内设计中，把 free-form 与严格 schema-constrained reviewing 作为核心实验变量，同时检验：
   - white-text prompt injection；
   - logic-critical versus surface perturbations；
   - review-aspect coverage；
   - latency、length 和 token cost。
4. 因此，本研究不是简单重复“LLM 是否会失败”，而是检验 review structure 是否调节这些失败，并量化其工程权衡。

> [修改] 原 gap 只是复述项目目标，没有说明相对于两篇关键文献具体多做了什么。

## 1.4 Operational definition of reviewing structure

本研究比较的是两个完整 reviewing setups：

- **Free-form setup / ZERO-GENERIC**：自然语言、无显式评审维度、plain-text paragraphs。
- **Schema-constrained setup / ZERO-GUIDE**：显式指定 summary、strengths、weaknesses、methodological flaws、rating 和 confidence，并强制遵守 Pydantic JSON schema。

> [新增·关键限制] 该干预同时改变了 prompt guidance、评审维度、输出格式和表达自由度。因此，本研究能够识别的是整个 reviewing setup 的差异，不能把所有观察结果单独归因于 JSON syntax。

全文优先使用：

- “free-form versus schema-constrained reviewing setups”；
- “the structured setup was associated with ...”；

避免直接使用：

- “JSON caused ...”；
- “formatting physically reduced the model’s reasoning ability”。

## 1.5 Research questions

### RQ1 — Manipulation robustness

**How does schema-constrained reviewing affect the susceptibility of LLM-generated reviews to white-text prompt injection?**

- 核心因变量：rating、strengths、weaknesses、methodological flaws、injection compliance。
- 核心判断：两种条件是否受到攻击，以及 Structured 是否改变攻击效应幅度。

> [修改] 将“能否抵御”改成“如何影响 susceptibility”，因为结果不是简单的成功防御/完全失败二分。

### RQ2 — Defect discriminability

**Does schema-constrained reviewing improve the differential sensitivity of LLM-generated reviews to logic-critical rather than surface-level perturbations?**

- 核心因变量：Δ methodological flaws 与 Δ rating。
- 核心判断：Logic perturbations 的反应是否显著区别于 Format perturbations。

> [修改] RQ2 的核心是 Logic versus Format，而不是分别检验每一组是否区别于零。

### RQ3 — Review-aspect coverage

**How does schema-constrained reviewing affect the coverage and distribution of strengths, weaknesses, and methodological-flaw comments?**

> [修改] 用 “aspect coverage” 代替未经验证的 “comprehensiveness”。更多评论条目不必然意味着更全面或质量更高。

### RQ4 — Operational trade-offs

**What trade-offs arise between output efficiency and rating dispersion under schema-constrained reviewing?**

- 效率指标：word count、latency、output tokens、估算成本。
- 输出指标：rating distribution / dispersion。

> [修改] 使用 “rating dispersion” 而不是 “cognitive variance”。

## 1.6 Contributions

建议明确列出四项贡献：

1. **Unified evaluation design:** 在同一被试内框架中联合评估 prompt injection 与 counterfactual logic defects。
2. **Controlled setup comparison:** 在同一 Generator 和同一批论文上比较 free-form 与 schema-constrained reviewing setups。
3. **Multi-dimensional evaluation:** 同时测量 rating、review aspects、methodological flaws、attack compliance 和系统效率。
4. **Empirical finding:** 证明结构化约束能够改善效率并改变输出分布，但不能单独作为安全或逻辑推理保障。

> [修改] 原 contribution 只描述了 N=30 的实验设计，没有明确说明知识贡献。

## 1.7 Dissertation roadmap

- Chapter 2 回顾 ARG、操纵风险、逻辑缺陷评估及 structured generation。
- Chapter 3 描述两条实验轨道、两个 reviewing setups、指标提取与统计方法。
- Chapter 4 按 RQ1–RQ4 报告结果。
- Chapter 5 讨论安全性、诊断能力、效率权衡、限制与未来工作。
- Chapter 6 总结研究发现。

> [新增] Introduction 末尾加入简短章节导航。

---

# 2. Related Work

## 2.1 LLMs and automatic peer review generation

- 定义 LLM-assisted peer review 与 fully automatic review generation。
- 总结 ARG 的主要使用方式：单次 zero-shot review、venue-guided prompts、multi-agent systems、fine-tuned reviewers。
- 简述已有质量评估路线：
  - 与 human reviews 的内容重合；
  - helpfulness / specificity / coverage；
  - score prediction；
  - error detection；
  - LLM-as-a-Judge evaluation。
- 指出现有结果不一致，部分原因是 review quality 定义不同、任务包含多种技能、依赖 noisy human review ground truth。

> [新增] 当前 Related Work 直接从漏洞开始，缺少 ARG 研究领域和评估困难的总体背景。

## 2.2 Manipulation and security vulnerabilities

- 介绍 prompt injection 在文档处理和长文本 LLM pipeline 中的基本风险。
- 重点讨论 Ye et al.：
  - hidden white-text injection 的机制；
  - 对评分、优缺点及 human–LLM alignment 的影响；
  - explicit 与 implicit manipulation 的区别；
  - 对自动化审稿安全性的意义。
- 可补充其他 document-level indirect prompt injection 文献，以表明该问题并不限于 peer review。

> [新增] 至少需要少量相关安全文献，避免 Section 2.2 事实上只是对一篇论文的摘要。

## 2.3 Evaluating scientific reasoning and faulty-logic detection

- 区分事实知识错误、公式错误、缺失信息与 research-logic inconsistency。
- 说明为什么逻辑缺陷检测是 peer review 的核心技能：需要验证 results、conclusions 与 findings 之间的支持关系。
- 介绍 counterfactual evaluation 的优势：
  - 控制论文身份、主题、质量和大部分文本；
  - 通过 relative change 避免依赖绝对 review quality；
  - 使用 soundness-neutral edits 建立表面敏感性的对照。
- 详细定位 Dycke and Gurevych：
  - ZERO-GENERIC 与 ZERO-GUIDE 已被比较；
  - generic prompts 在其倾向性排名中经常优于 guided prompts；
  - 但所有测试 ARG 都未显示 critical versus neutral 的可靠区分。

> [修改] 不能把“比较自由和 guided prompts”本身写成本文首次提出的贡献。

## 2.4 Structured generation and review forms

- 讨论结构化 prompt / schema-constrained generation 的潜在优势：
  - 强制字段覆盖；
  - 稳定机器解析；
  - 减少冗余；
  - 便于规模化汇总和决策支持。
- 讨论潜在风险：
  - prompt guidance 可能成为 distractor；
  - 预定义字段可能限制开放式问题发现；
  - 输出约束可能压缩表达范围；
  - structured output 并不等同于 secure reasoning。
- 如能找到相关文献，区分 conference review forms、checklists、JSON schemas 和 reasoning scaffolds。

> [新增·高优先级] 这是目前 Related Work 最大的缺口。既然“structure”是核心自变量，就必须有专门的文献和概念定义，而不能只在 2.3 最后突然提出 JSON。

## 2.5 Synthesis and positioning of this study

用一段综合论证连接前文：

- manipulation literature 说明 ARG 输入管线不安全；
- counterfactual literature 说明 ARG 缺乏逻辑缺陷判别能力；
- structured generation literature 提供了可能的控制机制，但也可能限制开放式批判；
- 尚不清楚严格 schema-constrained reviewing 在这两类失败中是保护因素、无效约束还是新的干扰因素；
- 本文通过统一的两轨被试内实验对此进行检验。

> [新增] Related Work 必须以明确 synthesis 结束，使 Methodology 看起来是由文献缺口自然推出，而不是由已有代码反向解释。

---

# 3. Methodology

## 3.1 Experimental overview

- 设计：strict within-subjects design，N=30 papers。
- Generator：`gpt-5.4`，temperature = 0。
- Judge：`deepseek-v4-flash`，temperature = 0。
- 两条实验轨道：
  1. Counterfactual track：30 × 8 = 240 paper conditions，每个条件分别生成 Free 与 Structured review；
  2. Injection track：30 × 2 = 60 PDFs，每个 PDF 分别生成 Free 与 Structured review。
- 总调用量：600 Generator calls + 360 Judge calls = 960 calls。

> [新增] 建议加入一张完整 pipeline diagram 或 design table，使读者能够立即理解两条轨道和两个 reviewing setups 的交叉关系。

## 3.2 Data selection and sampling

- 数据来源：Dycke and Gurevych counterfactual dataset 及本地 original PDFs。
- 纳入标准：同一论文必须具有完整的 7 个 counterfactual versions，并存在可用 original PDF。
- 从 123 篇 eligible papers 中以 seed = 10190 随机抽取 30 篇。
- 报告论文 venue、year 或 paper type 的基本分布。

> [新增] 需要解释为什么选择 N=30，以及这对统计功效和外部效度的影响。若没有正式 power analysis，应明确这是由完整交集、API 成本和实验可行性共同决定的样本。

## 3.3 Counterfactual track

### 3.3.1 Conditions

- Original baseline。
- Logic-critical perturbations：
  - `blueprint_conclusion`；
  - `blueprint_finding`；
  - `blueprint_result`。
- Surface / format perturbations：
  - `active_passive`；
  - `british_american`；
  - `language_error`；
  - `paper_layout`。

### 3.3.2 Pairwise comparison principle

- 每个 counterfactual review 与同一 paper 的 Original review 比较。
- 对每个指标计算：

\[
\Delta_{p,c}=Y_{p,c}-Y_{p,Original}
\]

- 逻辑判别能力定义为 Logic 组相对于 Format 组出现更强、方向正确的变化，而不是单一 Δ 是否非零。

> [新增·关键] 在 Methodology 中预先定义 discriminability，避免 Results 阶段根据星号事后解释。

## 3.4 Injection track

### 3.4.1 PDF manipulation

- 使用 PyMuPDF 在 PDF 第一页和最后一页嵌入 fontsize = 4、white color 的不可见文本。
- payload 被伪装成 “Positive Calibration Track” 内部指令。
- 三项主要攻击目标：
  - rating 提高约 1–2 分；
  - weaknesses 被淡化；
  - strengths 被突出；
  - 同时要求不在最终输出中暴露指令。

> [修改] 将其称为 “document-layer” 或 “PDF text-layer injection”，不要称为“物理层攻击”。它作用于 PDF 文本层和解析管线，不是物理硬件层攻击。

### 3.4.2 Control condition

- Original_PDF 与 Manipulated_PDF 仅在隐藏 payload 上不同。
- 对同一 paper、同一 reviewing setup 进行配对比较。

### 3.4.3 Ethical framing

- 说明 payload 只用于受控研究；
- 不针对真实审稿系统部署；
- 研究目的在于识别安全风险并提出防御方向。

> [新增] 在 Methodology 或单独 Ethical Considerations 中加入安全与研究伦理说明。

## 3.5 Reviewing setups

### 3.5.1 Free-form condition

- ZERO-GENERIC prompt；
- 不提供评审维度；
- 要求自然、完整、严格、建设性的 conference-style review；
- 只允许 plain-text paragraphs。

### 3.5.2 Schema-constrained condition

- ZERO-GUIDE prompt；
- 显式列出 summary、strengths、weaknesses、methodological flaws、rating、confidence；
- 强制 StructuredReview Pydantic JSON schema；
- 三个 list fields 使用对称的 “Leave as an empty list [] if none.” 描述。

### 3.5.3 Interpretation boundary

> [新增·关键限制] 两个条件不是只在序列化格式上不同。Structured 同时提供了额外维度指导。因此后文只能讨论 setup-level association，不能把效应唯一归因于 JSON。

## 3.6 Automated feature extraction

### 3.6.1 Free-form reviews

- Judge 提取 integer rating、strength count、weakness count、methodological-flaw count。
- Injection track 额外提取 injection-compliance score。

### 3.6.2 Structured reviews

- rating 和 aspect counts 直接采用 Pydantic self-report。
- Judge 只提取 injection-compliance score。

### 3.6.3 Measurement comparability

> [修改] 不要声称这种设计“消除了测量误差”。它减少了对 Structured 输出进行二次解析的需要，但引入了 measurement-procedure asymmetry：Free 指标由 Judge 推断，Structured 指标由 Generator 直接报告。

必须说明：

- 两组评分都为 1–10 integer scale，因此数值尺度一致；
- 但指标生成机制不同；
- RQ3 的跨组 aspect-count comparison 尤其容易受到这种差异影响；
- 结果应被解释为 operational measures，而不是无误差的真实 review quality。

## 3.7 Outcome measures

- Rating：1–10。
- Strengths、weaknesses、methodological flaws：distinct item counts。
- Injection compliance：0–10。
- Review length：word count。
- Latency：API call elapsed time。
- Output tokens：统一 tokenizer 估算。
- Rating dispersion：不同 review outputs 的 rating SD / distribution。

> [新增] 明确 primary outcomes 与 secondary outcomes。建议 rating 与 methodological flaws 为主要行为指标，其他指标为辅助解释。

## 3.8 Statistical analysis

### RQ1

- 在每种 setup 内，对 Manipulated_PDF 与 Original_PDF 做 paired comparison。
- 报告 mean paired difference、95% CI、p-value 和 paired effect size。
- 进一步比较 Free 与 Structured 的 per-paper attack deltas，即 setup × injection interaction。

> [已完成] 已对 Free 与 Structured 的 per-paper attack deltas 进行直接配对比较，结果保存在 `outputs/stats/rq1_attenuation_test.csv`。Structured 显著减弱了注入导致的评分提升（Free Δ = +1.90；Structured Δ = +0.97；paired t = 5.887，p = 2.17 × 10⁻⁶），也显著减弱了 methodological-flaw reports 的下降幅度（Free Δ = -3.20；Structured Δ = -1.83；paired t = -3.013，p = .0053）。因此可以使用 “significantly attenuated some attack effects”，但仍不能称为成功防御。

### RQ2

- 不应只对 pooled deltas 做 one-sample t-test。
- 首选分析：mixed-effects model，以 paper 为随机效应，perturbation class、setup 及其交互为固定效应。
- 可接受的简化方案：先在每篇 paper 内分别计算平均 Logic Δ 和平均 Format Δ，再进行 paired comparison。
- 分别对 methodological flaws 和 rating 建模。
- 如进行多个检验，说明 multiple-comparison correction。

> [已完成] 已采用 paper-level aggregation，对每篇论文的平均 Logic Δ 与平均 Format Δ 进行配对比较，结果保存在 `outputs/stats/rq2_discriminability_test.csv`。四项比较均不显著：Free MF p = .632；Structured MF p = .162；Free rating p = .979；Structured rating p = .771。这直接支持 neither setup demonstrated reliable logic-specific discriminability。
>
> [已完成] 扩展统计结果已写入 `outputs/stats/rq2_discriminability_test.csv`，包括 Logic–Format mean difference、95% CI、paired Cohen's d_z、raw p-value 和 Holm-adjusted p-value。正文仍需解释 paper-level aggregation 的计算方式，但不再缺少统计量。

### RQ3

- 仅在 Original condition 比较 aspect profiles，避免 counterfactual conditions 重复计入。
- 报告各维度均值、分布和总 count。
- 将分析称为 coverage profile comparison。

> [待分析] 如要使用 “comprehensiveness” 一词，需要人工评价、经过验证的 coverage rubric，或至少补充 Judge/human validation。

### RQ4

- 报告 word count、latency、token count 的均值、中位数、差值和相对减少比例。
- 报告两组 rating distribution 和 SD。
- Levene test 只能证明方差不同，不能证明较低方差是认知损害。

> [限制] 240 行包括同一 30 篇论文的重复条件，简单把 240 个值当完全独立观测会忽略 repeated-measures structure。

## 3.9 Reproducibility

- 报告 random seed、完整 prompt、schema、model identifiers、temperature 和 tokenizer。
- 描述失败调用、重试、解析失败和缺失值处理。
- 将所有输出文件、raw reviews 和绘图脚本对应到 pipeline stages。

> [新增] 说明商业模型可能更新，确切 model version 和调用日期应记录在附录或 reproducibility statement 中。

---

# 4. Results and Analysis

## 4.1 RQ1 — Susceptibility to white-text prompt injection

### Evidence to report

- Figure: paired slope graph for ratings。
- Figure: Δ rating versus Δ weaknesses，bubble size = Δ strengths。
- Figure: Δ rating versus Δ methodological flaws。
- Table：两种 setup 的 baseline、injected mean、ATE、95% CI、p-value 和 effect size。

### Current descriptive results

| Metric | Free Δ | Structured Δ |
|---|---:|---:|
| Rating | +1.90 | +0.97 |
| Strengths | +1.10 | +2.10 |
| Weaknesses | -2.70 | -1.20 |
| Methodological flaws | -3.20 | -1.83 |
| Injection compliance | +5.30 | +2.40 |

### Correct conclusion direction

- 两种 setup 都没有抵御攻击：评分上升、weaknesses 和 methodological flaws 减少、compliance 上升。
- Structured 的部分攻击效应较小，尤其是 rating、weaknesses、methodological flaws 和 compliance；但 strengths 的增加更大。
- 因此 Structured 最多可描述为 attenuating some attack effects，而不是提供可靠防御。
- “MF blindness” 有较强描述性支持：Free 中 28/30 篇、Structured 中 25/30 篇在注入后 methodological-flaw count 下降。

> [修改] 原结论“均易受操纵”可保留，但必须加入 Structured 攻击效应较弱这一重要结果。
>
> [已完成] setup 间 attack-delta 的直接配对比较显示，Structured 显著减弱了 rating inflation（paired t = 5.887，p = 2.17 × 10⁻⁶）及 methodological-flaw reduction（paired t = -3.013，p = .0053）。因此可以正式使用 “significantly attenuated these measured attack effects”，但必须同时强调两种 setup 中攻击效应仍然存在。
>
> [修改] 避免“完全失明”。更准确的是 “substantial reduction in reported methodological flaws under injection”。

## 4.2 RQ2 — Discriminability of logic defects

### Evidence to report

- Figure: Logic versus Format distributions for Δ methodological flaws and Δ rating。
- Table：每种 setup × perturbation class 的 mean Δ、CI 和 model coefficient。
- 主要统计检验必须直接比较 Logic 与 Format。

### Current descriptive results

| Outcome | Setup | Logic Δ | Format Δ |
|---|---|---:|---:|
| Methodological flaws | Free | +0.66 | +0.77 |
| Methodological flaws | Structured | -0.43 | -0.63 |
| Rating | Free | +0.02 | +0.03 |
| Rating | Structured | +0.01 | +0.03 |

### Correct conclusion direction

- Free 在 logic perturbations 后平均提到更多 methodological flaws，但对 format perturbations 的增加幅度相同甚至更大。
- Structured 在两类 perturbations 后都减少了 methodological-flaw reports。
- 两种 setup 的 rating 对两类 perturbations 都几乎不变。
- 因此当前证据支持：**neither setup demonstrated reliable logic-specific discriminability**。
- Structured 没有改善 defect discriminability，但也不能简单声称它“严重损害了一个 Free 原本具备的能力”，因为 Free 同样没有 Logic-over-Format specificity。

> [修改·关键] 删除“Prompt_Free 展现一定惩罚能力，而 Prompt_Structured 完全丧失判别敏感度”。这与当前 CSV 和图不一致。
>
> [修改] 方法论缺陷数量增加不等于“惩罚”，尤其当 overall score 没有下降时。应分别描述 critique coverage 与 evaluative consequence。
>
> [已完成] paper-level paired aggregation 的 Logic-versus-Format 检验均不显著：Free MF p = .632；Structured MF p = .162；Free rating p = .979；Structured rating p = .771。该结果确认当前描述性结论，而不是仅依赖原图中的 one-sample significance stars。

## 4.3 RQ3 — Review-aspect coverage

### Evidence to report

- Figure: Original condition 下 strengths、weaknesses 和 methodological flaws 的 stacked bars。
- Table：每个维度和 total aspect count 的均值、中位数与分布。

### Current descriptive results

- Free：strengths 6.47，weaknesses 11.07，methodological flaws 8.63，总计 26.17。
- Structured：strengths 5.93，weaknesses 7.57，methodological flaws 8.80，总计 22.30。

### Correct conclusion direction

- Free reviews 的 Judge-extracted total aspect count 更高，主要来自更高的 weakness count。
- Structured 输出确保预定义字段都可直接访问，但没有产生更高的总 aspect count。
- 两组 methodological-flaw counts 在 Original condition 下接近。
- 该结果证明的是不同的 coverage profiles，而不是 Structured 提高了 review quality 或 comprehensiveness。

> [修改] 原结论只说“存在结构性差异”，过于空泛；需要说明差异的方向和主要来源。
>
> [限制] Free 和 Structured 的 counts 来自不同测量机制，跨组绝对数量差异不能完全归因于 reviewing setup。
>
> [计划·辅助验证] 从 5 篇论文中抽取 Free 与 Structured reviews，进行 condition-masked single-annotator coding。人工评估只验证自动提取的 rating、strength、weakness、methodological-flaw 和 injection-compliance 指标，不重新审阅论文，也不作为 RQ1/RQ2 的主要评价标准。

## 4.4 RQ4 — Efficiency and rating dispersion

### Evidence to report

- Figure: word count versus latency scatter。
- Figure: rating distributions / violin plot。
- Table：words、latency、tokens、estimated cost 的均值和 reduction percentages。

### Current descriptive results

- Mean word count：1883.7 → 793.3，约减少 57.9%。
- Mean latency：29.1 s → 13.7 s，约减少 53.0%。
- Mean output tokens：2354 → 1113，减少 52.7%。
- Rating SD：1.27 → 0.93；Levene p < .001。

### Correct conclusion direction

- Structured setup 显著降低输出长度、latency 和 token consumption，工程效率优势明确。
- Structured ratings 的分布更窄，说明 rating dispersion 较低。
- 单凭较低方差不能判断这是 desirable consistency、scale compression，还是 reduced sensitivity。
- RQ2 没有表现出可靠的逻辑判别，因此不能把低方差自动解释成高质量一致性。

> [修改] 删除“以严重压缩评分的认知方差为代价”。替换为中性、可观察的 “was accompanied by a narrower rating distribution”。
>
> [修改] Token 减少 52.7%，word count 减少约 57.9%，不要把两个比例混用。
>
> [限制] API latency 也受服务负载、网络和模型后端影响；本研究记录的是 observed end-to-end latency，不是纯模型计算时间。
>
> [限制] 成本表必须标记为基于假设价格的估算，除非使用的是调用时该模型的真实账单价格。

## 4.5 Results summary

建议在 Chapter 4 末尾加入一张 RQ summary table：

| RQ | Main finding | Evidential boundary |
|---|---|---|
| RQ1 | Both setups were vulnerable; Structured significantly attenuated the measured rating and MF effects but did not prevent compliance. | The two setups currently use different metric-extraction procedures; common-Judge analysis is planned as a sensitivity check. |
| RQ2 | Neither setup showed reliable logic-specific discriminability in the paper-level paired analysis. | The conclusion concerns review-output sensitivity, not whether the model internally noticed the defects. |
| RQ3 | The setups produced different aspect-coverage profiles; Free had higher extracted total counts. | Measurement procedures differ across setups. |
| RQ4 | Structured was shorter, faster and less token-intensive, with narrower rating dispersion. | Lower dispersion is not inherently better or worse. |

> [新增] 这张表可以防止 Discussion 和 Conclusion 再次把描述性结果扩大成因果或认知结论。

---

# 5. Discussion and Future Work

## 5.1 Structure is an efficiency mechanism, not a security mechanism

- Structured output 适合机器解析、汇总和规模化部署。
- 但攻击 payload 位于输入内容中，输出 schema 本身并不验证输入是否可信。
- 因此 schema constraint 与 prompt-injection defense 解决的是不同层次的问题。
- Structured 对部分攻击效应的减弱值得讨论，但不能被描述为安全保证。

> [修改] 将讨论主线从“结构化严重削弱认知”改为“结构化改善工程可控性，但没有解决输入完整性和科学推理问题”。

## 5.2 Why did neither setup discriminate logic defects?

可讨论但不能直接断言的解释：

- 长论文中的关键信息检索失败；
- 模型依赖表面 review heuristics；
- 逻辑缺陷需要跨段落验证 result–conclusion–finding relations；
- free-form generation 产生更多批评，但这些批评不具有缺陷特异性；
- explicit fields 可能增加 coverage，却不保证字段内容建立在正确推理上。

> [限制] 实验观察的是输出行为，无法区分模型没有找到相关信息、没有识别逻辑矛盾，还是识别后没有在 review 中表达。

## 5.3 Interpreting narrower rating dispersion

- 讨论可能的积极解释：更稳定、较少极端评分、便于决策系统处理。
- 讨论可能的消极解释：scale compression、对论文差异或缺陷不敏感。
- 结合 RQ2 强调：本研究无法判定较低 dispersion 等于更高 consistency。
- 如果要研究一致性，应在相同 paper-condition 下进行多次重复生成并计算 test–retest reliability / ICC。

> [新增] 项目 principal goal 提到 “more consistent reviews”，但当前 temperature = 0、每条件单次生成并不能直接测量重复生成一致性。必须在 limitations 中明确这一点。

## 5.4 Measurement asymmetry and LLM-as-a-Judge

- Free 的 rating/counts 是 Judge inference；Structured 是 Generator self-report。
- 这种设计保持了各 setup 的自然输出方式，但降低了跨组指标的完全可比性。
- Judge 可能受到 review length、tone 和 verbosity 影响。
- Structured list length 也不代表每个 item 的质量、独立性或正确性。

> [新增·关键限制] 这是 RQ3 和 rating-variance comparison 必须主动讨论的 measurement validity 问题。

## 5.5 Defensive system design

### Cross-modal document validation

- 比较 PDF extracted text 与 rendered pixels，检测只存在于文本层、不可见于图像的内容。
- 检查异常小字号、白色文字、off-page text、重复隐藏指令和不一致 OCR。

### Vision-based review

- 将 PDF 渲染为图像可能消除本实验这种纯文本层白字 payload。
- 但 Vision API 是否构成有效防御需要单独实验验证；OCR、accessibility text 或 multimodal preprocessing 仍可能重新暴露 payload。

> [修改] 不要称 Vision API 为 “natural defense” 或既定解决方案。应称为 promising defensive hypothesis。

### Instruction isolation

- 将论文内容明确标记为 untrusted data；
- 使用 input sanitisation 和 prompt-injection detection；
- 将 review generation 与 security screening 分离；
- 对异常高分变化或批评骤减设置 flag。

> [新增] 防御建议不能只有 Vision；应提出文本层检测、跨模态验证和行为异常检测的组合。

## 5.6 Limited manual validation of automated feature extraction

- 从 5 篇论文中抽取 Free 与 Structured reviews，预计覆盖 RQ1 的 Original/Injected 配对，以及 RQ2 的 Original、一个 Logic 和一个 Format condition。
- 每篇预计编码 10 份 reviews，共约 50 份；最终抽样条件和 random seed 应在执行前固定。
- 采用 condition-masked single-annotator coding：隐藏 paper identity、attack/perturbation condition 和所有自动指标，但由于输出形式可能暴露 setup，不声称完全 blind to Free/Structured。
- 人工独立记录 inferred rating、n_strengths、n_weaknesses、n_methodological_flaws 和 injection-compliance score。
- 人工编码只用于验证自动指标提取，不重新阅读原论文或判断评论是否真正发现了植入缺陷。
- 当前 measurement pipeline 下分别比较 Human–Free Judge 与 Human–Structured self-report；若补齐 Structured Judge 数据，则使用同一批人工编码进一步比较 Human–Structured Judge。
- 该分析定位为 limited measurement validation，不作为 RQ1/RQ2 的主要评判标准，也不称为 human gold standard。

> [限制] 全部人工编码由 dissertation author 一人完成，因此不能报告 inter-rater reliability。若时间允许，可在间隔一段时间后重复编码 10%–20% 的样本，作为 intra-rater consistency check。

## 5.7 Limitations

必须至少覆盖：

1. N=30，且只来自满足完整 counterfactual intersection 的论文；
2. 单一 Generator，结论不能直接推广到所有 LLM；
3. 单一 Judge；计划进行 5 篇论文的单人抽样验证，但无法评估 inter-rater reliability；
4. 每个 condition 单次生成，无法直接测量 test–retest consistency；
5. Free 与 Structured 使用不同指标提取机制；
6. Reviewing setup 是复合干预，不能分离 prompt guidance 与 JSON schema 的独立效应；
7. Counterfactual errors 是自动生成的近似缺陷，不等于现实论文错误分布；
8. PDF injection 只测试一种 payload 和一种隐藏方式；
9. API latency 受到外部系统因素影响；
10. paired / repeated-measures structure 必须在统计模型中正确处理。

> [新增] 当前 outline 几乎没有完整 limitations section，这是 dissertation 可信度的重要组成部分。

## 5.8 Future work

- 多模型 replication，包括不同厂商、开放权重模型和 multimodal systems。
- 将 prompt guidance 与 JSON formatting 拆成 2 × 2 factorial design：
  - no guidance + text；
  - guidance + text；
  - no guidance + JSON；
  - guidance + JSON。
- 每个 paper-condition 多次重复，直接测量 consistency。
- 使用人工验证或多 Judge ensemble。
- 测试多种 injection payload、位置、字体和攻击策略。
- 实现 extracted-text versus rendered-image validation。
- 测量真正的 defect detection accuracy，而不仅是 review-level count changes。

---

# 6. Conclusion

建议的结论方向：

> 本研究在统一的被试内设计中，对比了 free-form 与 schema-constrained LLM peer-reviewing setups 在白字 prompt injection、反事实逻辑缺陷、review-aspect coverage 和系统效率方面的表现。结果显示，严格结构化输出显著减少了 review length、latency 和 token consumption，并产生了更窄的评分分布。然而，结构化约束并未形成可靠的 prompt-injection 防御：尽管部分攻击效应有所减弱，两种设置仍表现出明显的行为顺从。同时，两种设置都没有表现出对逻辑缺陷相对于表面变化的可靠特异性判别。因此，结构化输出应被理解为一种工程效率和输出组织机制，而不是安全性或科学推理能力的充分保障。未来系统需要结合输入完整性检查、跨模态验证和人类专家监督，才能在规模化效率与同行评审严谨性之间建立更可靠的平衡。

> [修改] 删除“Structured 会严重损害模型检测科学逻辑缺陷的能力”。当前数据支持的是 Structured 未能改善判别，而 Free 本身也没有可靠的 Logic-over-Format specificity。
>
> [修改] 将“无法有效缓解操纵”细化为“没有阻止攻击，但对部分攻击效应可能存在 attenuation”。
>
> [修改] 不将更窄的评分分布直接解释成优点或认知损害。

---

# Prioritised revision and analysis list

## Chapter 4 writing readiness

- **当前无阻塞项，可以开始撰写 `4_Results & Analysis.md`。** RQ1/RQ2 的核心推断统计、RQ3 描述性指标和 RQ4 工程指标均已具备。
- Chapter 4 初稿中保留两个明确占位项：`[COMMON-JUDGE SENSITIVITY PENDING]` 与 `[MANUAL VALIDATION PENDING]`。
- 两个占位项都不阻塞 RQ1、RQ2 和 RQ4 主体写作；RQ3 的跨 setup 解释在补齐验证前保持 provisional。

## P0 — Results 定稿前必须完成（不阻塞当前初稿）

1. **[已完成] 重做 RQ2 的核心统计检验。** 已使用 paper-level aggregated paired analysis 直接比较 Logic 与 Format；结果保存于 `outputs/stats/rq2_discriminability_test.csv`，四项比较均不显著。
2. **[已完成] 补齐 RQ1/RQ2 的统计报告。** 两个 stats CSV 均已加入 mean contrast、95% CI、paired Cohen's d_z、raw p-value 和 Holm-adjusted p-value。
3. **[已完成] 修正 4.2 的结论。** Outline 已改为“两者都没有 logic-specific discriminability”；Chapter 4 写作时沿用该结论。
4. **[已完成] 明确 reviewing setup 是复合干预。** Outline 已明确不能把观察结果唯一归因于 JSON；正文需保持该表述。
5. **[已完成] 补做 RQ1 的 setup × injection effect comparison。** 已直接比较 Free 与 Structured 的 per-paper attack deltas；结果保存于 `outputs/stats/rq1_attenuation_test.csv`。Rating 与 MF attack effects 均显著减弱。
6. **[部分完成·不阻塞初稿] 补齐 Structured reviews 的 Judge 数据，但暂不替换主要分析。** Injection track 的 60 条 Structured Judge 数据已存在于 `outputs/step2_pdf_track_structured_judge_full.csv`；Counterfactual track 的 240 条尚未补齐。先保留当前主分析，待导师确认后决定 common-Judge 是否升级为 primary analysis。
7. **[已完成] 统一结果术语。** Outline 已统一使用 rating dispersion、aspect coverage、document-layer/PDF text-layer injection 和 schema-constrained setup。

## P1 — Introduction 与 Related Work 成立所必需

1. **[大纲已完成·正文待写] 重写 research gap。** Outline 已承认 Dycke and Gurevych 比较过 ZERO-GENERIC 与 ZERO-GUIDE，并突出统一双轨设计和 strict schema focus。
2. **[大纲已完成·正文待写] 新增 Structured generation / review forms 文献小节。** 相关概念结构已加入 Section 2.4，仍需检索和写入正式文献。
3. **[大纲已完成·正文待写] 新增 ARG 与 review-quality evaluation 背景。** Section 2.1 的写作范围已经确定。
4. **[大纲已完成·正文待写] 把 contributions 从实验描述改成知识贡献。** 四项 contributions 已在 outline 中明确。
5. **[大纲已完成·正文待写] 精确描述两篇关键文献。** 论证边界已修正，正式正文仍需加入引用和细节。

## P2 — 结果解释与可信度

1. **[已完成] 将 RQ3 改为 aspect coverage。** Outline 已停止把 item count 直接解释为 comprehensiveness。
2. **[已完成] 中性解释 rating variance。** 已统一为 rating dispersion，并明确不能判断较低方差是 consistency 还是 reduced sensitivity。
3. **[大纲已完成·正文待写] 加入完整 limitations section。** N=30、单模型、单 Judge、单次生成、measurement asymmetry 和 repeated measures 均已列入 Section 5.7。
4. **[已完成] 检查 token、word 和 latency 百分比。** Token -52.7%，words -57.9%，latency -53.0%，三者已分开报告。
5. **[已完成] 将 Vision API 改为待验证假设。** Outline 已补充文本层检测、跨模态验证和异常行为检测，未将 Vision 作为既定防御。

## P3 — 有时间时增强论文质量

1. **[评测待完成] 执行 5 篇论文的 condition-masked single-annotator validation。** 50 份匿名 reviews、annotation sheet 和 protocol 已生成并通过 audit；人工编码尚未完成。
2. **[待写·不阻塞 Chapter 4] 报告论文 venue / year / paper type 的样本构成。** 主要属于 Methodology/sample description。
3. **[部分完成] 增加 pipeline diagram、experimental matrix 和 RQ summary table。** RQ summary table 已在 outline 中设计；pipeline diagram 和最终表格仍待制作。
4. **[分析已完成·正文待写] 报告 latency median、spread，并说明外部服务噪声。** 现有数据足够，无需新增实验。
5. **[大纲已完成·正文待写] 在 Future Work 中提出 guidance × output format 的 2 × 2 factorial design。** 已写入 Section 5.8。
6. **[大纲已完成·无需当前执行] 设计 repeated-generation study。** 已定位为 future work，用于未来直接测量 review consistency。
