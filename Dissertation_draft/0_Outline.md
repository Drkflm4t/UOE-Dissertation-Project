# Dissertation Master Outline

## Working title

### Recommended

**Structured LLM Peer Review: Manipulation Robustness, Reasoning Sensitivity, and Operational Trade-offs**

### Alternatives

- **Structured Reviewing as a Control Layer for LLM-Based Peer Review**
- **Evaluating Structured LLM Reviewers under Manipulation and Faulty Research Logic**
- **From Free-Form to Structured Review: Robustness, Coverage, and Efficiency in Automatic Peer Review**

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
- **[限制]**：必须在正文中主动承认的证据边界。
- **[可保留]**：现有方向与证据基本一致，可据此展开正文。

---

# Central argument of the dissertation

现有研究已经分别证明：LLM 自动审稿系统容易受到隐藏文本注入操纵，也不能可靠地区分论文中的逻辑缺陷与意义不变的表面变化。然而，严格的 schema-constrained reviewing setup 是否会改变这两类失败，以及这种约束会带来哪些内容和工程层面的代价，仍缺少统一、受控的实证研究。

本研究在同一批论文、同一生成模型和严格的被试内设计下，对比 free-form 与 schema-constrained 两种审稿设置。结果应围绕以下中心论点组织：

> **Schema-constrained reviewing provides measurable attenuation of selected manipulation effects and substantial gains in operational concision, while targeted audits show that reliable logic-defect detection requires complementary verification beyond output structure alone.**

全文围绕以下已建立的认识组织：

1. 两种审稿设置都容易受到白字 prompt injection；
2. Structured 条件下部分攻击效应较小，但攻击没有被阻止；
3. 两种设置都没有可靠地区分逻辑扰动和表面扰动；
4. Structured 输出显著更短、更快、token 消耗更低；
5. Common-measure coding 显示两种 setups 的 total recorded coverage 相当，Human benchmark 进一步表明 Structured 的篇幅明显更接近实际 human reviewing；
6. Structured 的评分分布更窄，作为独立的 distributional property 报告，而不让其承担 efficiency/concision contribution 的证明责任；
7. Target-defect audit 将 RQ2 的 aggregate null 分解为 target omission 与 critique–rating decoupling，为未来 reasoning-verification design 提供具体方向。

## Evidence-framing framework（后续各章统一遵循）

1. **先讲发现和贡献。** 每个 RQ 先用 primary analysis 回答研究问题，明确 effect direction、magnitude 与实际意义；不用 limitations 开场，也不让技术性边界遮住核心结果。
2. **再用多源证据加强论证。** 自动指标、common auxiliary coding、qualitative cases 与 human-reference benchmark 承担不同作用；它们用于互相校准、解释机制并检验结论稳健性。
3. **把 measurement asymmetry 写成已被主动处理的设计问题。** Primary analysis 遵循实验中已实施的 Free-Judge 与 Structured-native measurement procedures；5-paper common manual coding 为两种 outputs 提供共同 rubric，**partially bridges and calibrates the measurement gap**。因此应同时报告 primary pipeline 与已完成的 measurement triangulation，而不只列出 asymmetry。
4. **根据证据强度分层下结论。** Primary statistics 与 auxiliary evidence 方向一致时，可清晰使用 `supported` 或 `convergent evidence`；仅有 exploratory evidence 时使用 `consistent with`；真正无法区分的解释才保留为 future work。
5. **Limitations 与贡献保持比例。** 不回避重要限制，但也不将已通过验证、triangulation 或谨慎措辞处理的小缺陷写成整个项目失效。剩余的外部效度与扩展性问题放入 Limitations/Future Work。

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

建议明确列出六项贡献：

1. **Unified evaluation design:** 在同一被试内框架中联合评估 prompt injection 与 counterfactual logic defects。
2. **Controlled setup comparison:** 在同一 Generator 和同一批论文上比较 free-form 与 schema-constrained reviewing setups。
3. **Multi-dimensional evaluation:** 同时测量 rating、review aspects、methodological flaws、attack compliance 和系统效率。
4. **Measurement triangulation:** 在保留两条 operational pipelines 的同时，使用 condition-masked common coding 校准 Free-Judge 与 Structured-native measures，区分可复现的行为结果与 measurement-sensitive aspect counts。
5. **Mechanism-level diagnosis:** 使用 target-defect audit 与 count-granularity inspection，将 aggregate patterns 分解为 omission、critique–rating decoupling 和 measurement-sensitive segmentation。
6. **Empirical insight:** 证明 Structured setup 能够减弱部分 injection effects，并以更接近 human-review practice 的篇幅实现 comparable common-coded coverage；同时识别需要 input-integrity 与 reasoning-verification modules 补充的风险。

> [修改] 原 contribution 只描述了 N=30 的实验设计，没有明确说明知识贡献。

## 1.7 Dissertation roadmap

- Chapter 2 回顾 ARG、操纵风险、逻辑缺陷评估及 structured generation。
- Chapter 3 描述两条实验轨道、两个 reviewing setups、指标定义与收集、evidence triangulation 和统计方法。
- Chapter 4 按 RQ1–RQ4 完成 quantitative result、manual/human validation、case mechanism analysis 与 RQ conclusions。
- Chapter 5 将这些结论与 manipulation/faulty-reasoning prior work 对话，提出 layered-system implications，并集中界定 scope 与 future work。
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

**章节职责：** 说明 inputs、generation、measurement、aggregation 和 validation procedures，使 Chapter 4 的每项结论均可复现。Methodology 不提前报告结果，也不按代码 Step 1–5 写成实验日志。

## 3.1 Experimental overview

- 设计：strict within-subjects design，N=30 papers。
- Generator：`gpt-5.4`，temperature = 0。
- Judge：`deepseek-v4-flash`，temperature = 0。
- 两条实验轨道：
  1. Counterfactual track：30 × 8 = 240 paper conditions，每个条件分别生成 Free 与 Structured review；
  2. Injection track：30 × 2 = 60 PDFs，每个 PDF 分别生成 Free 与 Structured review。
- 总调用量：600 Generator calls + 360 Judge calls = 960 calls。

加入一张完整 pipeline diagram，使读者立即理解两条 tracks、两个 reviewing setups、operational metrics 与 auxiliary evidence 的关系。

## 3.2 Data selection and sampling

- 数据来源：Dycke and Gurevych counterfactual dataset 及本地 original PDFs。
- 纳入标准：同一论文必须具有完整的 7 个 counterfactual versions，并存在可用 original PDF。
- 从 123 篇 eligible papers 中以 seed = 10190 随机抽取 30 篇。
- 报告论文 venue、year 或 paper type 的基本分布。

说明 N = 30 来自完整 counterfactual/PDF intersection 后的固定随机样本；外部推广范围集中放在 Chapter 5。

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

将 Free/Structured 明确定义为两个完整 reviewing setups：Structured 同时改变 dimension guidance、output organisation 和 machine-readability。本文关注的就是采用该完整 setup 的实际效果；若要分离 JSON 与 guidance，作为 Chapter 5 的 factorial future work。

## 3.6 Automated feature extraction

### 3.6.1 Free-form reviews

- Judge 提取 integer rating、strength count、weakness count、methodological-flaw count。
- Injection track 额外提取 injection-compliance score。

### 3.6.2 Structured reviews

- rating 和 aspect counts 直接采用 Pydantic self-report。
- Judge 只提取 injection-compliance score。

### 3.6.3 Measurement comparability

该设计首先保留了两种 output formats 的实际消费方式：Free prose 需要 Judge 进行二次提取，Structured output 则直接提供 machine-readable fields。两组 rating 均使用 1–10 integer scale，从而对齐了数值尺度。

研究进一步用 5-paper condition-masked common coding 对 Free 和 Structured texts 应用同一 rubric，为两种 operational measures 提供共同参照。这一步**主动校准并部分弥合了 measurement gap**：它验证了部分核心方向，同时识别出 weakness count 对 segmentation/procedure 更敏感。因此 RQ3 不是只依赖不同测量源的绝对 count，而是结合 primary operational comparison 与 common-measure triangulation 下结论。

正文仍应透明说明 Free 与 Structured 的指标生成机制不同，但该信息应作为读者理解 operational pipeline 和 triangulation design 的一部分，而不是否定所有跨 setup 分析。

## 3.7 Outcome measures

- Rating：1–10。
- Strengths、weaknesses、methodological flaws：distinct item counts。
- Injection compliance：0–10。
- Review length：word count。
- Latency：API call elapsed time。
- Output tokens：统一 tokenizer 估算。
- Rating dispersion：不同 review outputs 的 rating SD / distribution。
- Human benchmark：matched Original papers 的 paper-level mean word count，以及 author-coded aspect profile。

在本节同时定义每个 metric 的概念意义、scale、measurement source、distinct-item rule 和 analysis use。Primary outcomes 为 RQ1 rating/MF/compliance 与 RQ2 rating/MF contrasts；RQ3/RQ4 metrics 用于 coverage 与 operational analysis。

## 3.7A Evidence triangulation

- 5-paper / 50-review condition-masked single-annotator common coding：对 Free 与 Structured texts 使用同一 rubric。
- Post-unblinding 5-case target-defect audit：记录 target mention、correct interpretation 和 rating consequence。
- Count-granularity inspection：检查 segmentation/category allocation 对 count gap 的贡献。
- Human benchmark：11 篇 matched papers、45 份 eligible public primary reviews；同一 paper 内先聚合为 paper-level mean。
- Methodology 描述选择和编码规则；Chapter 4 报告实际 trends、案例和结论。

## 3.8 Statistical analysis

### RQ1

- 在每种 setup 内，对 Manipulated_PDF 与 Original_PDF 做 paired comparison。
- 报告 mean paired difference；对直接支撑 attenuation 的 rating 与 MF 报告 Free–Structured contrast、95% CI 和 Holm-adjusted p-value。
- 进一步比较 Free 与 Structured 的 per-paper attack deltas，即 setup × injection interaction。
- Methodology 不报告实际结果数值；这些内容统一放入 Chapter 4 Table 4.1。

### RQ2

- 不应只对 pooled deltas 做 one-sample t-test。
- 最终分析：先在每篇 paper 内分别计算平均 Logic Δ 和平均 Format Δ，再进行 paired comparison。
- 分别对 methodological flaws 和 rating 建模。
- 四个 Logic–Format tests 使用 Holm correction；主文报告 Logic Δ、Format Δ、raw-scale contrast、95% CI 和 Holm p，不重复 `t`、raw p 或 `d_z`。

### RQ3

- 仅在 Original condition 比较 aspect profiles，避免 counterfactual conditions 重复计入。
- 报告各维度均值、分布和总 count。
- 将分析称为 coverage profile comparison。

- Operational profiles 与 5-paper common coding、11-paper Human external profile 联合解释；统一使用 `review-aspect coverage`，不把 count 自动等同于 quality。

### RQ4

- 全部240 rows报告 word count、latency、token count 的核心描述统计和相对减少比例。
- 报告两组 rating distribution 和 SD。
- Levene test 只能证明方差不同，不能证明较低方差是认知损害。
- Human benchmark 只使用11篇 matched Original papers，以 paper-level mean 为单位报告 N、mean、median 和 range。

## 3.9 Reproducibility

- 报告 random seed、完整 prompt、schema、model identifiers、temperature 和 tokenizer。
- 描述失败调用、重试、解析失败和缺失值处理。
- 将所有输出文件、raw reviews 和绘图脚本对应到 pipeline stages。

> [新增] 说明商业模型可能更新，确切 model version 和调用日期应记录在附录或 reproducibility statement 中。

---

# 4. Results and Analysis

**章节职责：** 本章是全文的主要 empirical argument。每个 RQ 按 `quantitative pattern → triangulation → qualitative/case mechanism → RQ answer` 完成论证；数据直接支持的 target omission、critique–rating decoupling、segmentation sensitivity 和 criticism retention 均在本章分析。

## 4.1 RQ1 — Susceptibility to white-text prompt injection

### Evidence to report

- Figure: paired slope graph for ratings。
- Figure: Δ rating versus Δ weaknesses，bubble size = Δ strengths。
- Table：各 metric 的 Free/Structured Δ；rating 与 MF 额外报告 setup contrast、95% CI 和 Holm p。

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
- Structured 的部分攻击效应较小，尤其是 rating、methodological flaws 和 compliance；automatic weakness reduction 也较小，但该维度未在 common coding 中可靠复现。Strengths 的增加则在 Structured 中更大。
- 因此 Structured 最多可描述为 attenuating some attack effects，而不是提供可靠防御。
- MF reporting 的 Free reduction 更大，由 Table 4.1 和方向性数量表达；不再使用单独 scatter/trend line 或 “MF blindness” 标签。

Setup 间 attack-delta 的直接配对比较显示，Structured 显著减弱了 rating inflation 及 methodological-flaw reduction；common coding 与 matched case 进一步显示其在受到攻击时保留更多 substantive criticism。RQ1 的主结论为 **measurable partial attenuation**，并由 Chapter 5 将 complete defence 定位为 input-integrity layer 的任务。

正文用一个 matched Original/Manipulated × Free/Structured case 说明 Structured mandatory fields 仍保留 substantive criticism。该例只是 illustrative evidence，不作 prevalence 或单一因果机制判断。

## 4.2 RQ2 — Discriminability of logic defects

### Evidence to report

- Figure: Logic versus Format distributions for Δ methodological flaws and Δ rating。
- Table：每种 setup × perturbation class 的 mean Δ、Logic–Format contrast、95% CI 和 Holm p。
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

Paper-level paired aggregation 的四项 Logic-versus-Format contrasts 均不显著，且 raw-scale contrasts 接近 0、confidence intervals 跨 0。图和主表统一报告 direct comparisons 与 Holm-adjusted p-values。

5 个 matched Logic cases 中，Structured 识别 2/5，Free 识别 3/5；没有可靠 setup difference。10 份 setup-specific reviews 中约一半遗漏 target，因此 **unreliable target-defect detection 是首要失败**。在 5 份 detected reviews 中仅 1 份降低 primary rating，critique–rating decoupling 作为次级机制解释已识别但未扣分的 cases。正文使用 Case 3：两种 reviews 均识别 claim–result contradiction，但 primary ratings 均未改变。

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

Original-only common coding 得到 Free total = 18.4、Structured total = 19.2，而 automatic totals 为 29.2 与 24.0。Automatic Free > Structured ordering 未被复现；weakness agreement 尤其差（Free MAE = 6.28，Structured MAE = 3.40）。Pair A 同时显示 segmentation 与 weakness/MF category-boundary sensitivity，但无 item-level Judge outputs，不能确认具体 double-counting mechanism。因此当前 RQ3 结论是 **measurement-sensitive aspect profiles**，而不是稳定的 Free total-coverage advantage。

Human external profile 基于11篇 matched papers、45份 author-coded reviews：Human 平均约7.73 aspects，而 matched Structured/Free operational profiles 分别约22.82/25.45。它不替代 common coding，但进一步说明高 item counts 不等同于 normal human-review coverage。与 common-coded comparable totals 和 Pair A 联合后，Free 的额外 counts 更符合 greater segmentation/granularity，而不是稳定的 substantive-coverage advantage。

## 4.4 RQ4 — Efficiency and rating dispersion

### Evidence to report

- Figure: matched Human/Structured/Free length grouped-bar panel + rating-distribution panel。
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

当前 ratings 已不再近乎全部为 7；简单 lexical overlap 与少量 text inspection 也不支持“Structured reviews 近乎相同而导致 compression”的简单假设。在 25 + 25 reviews 的共同 inferred-rating coding 下，Free SD = 1.12，Structured SD = 0.53，方向上支持 Structured rating distribution 较窄，但 remaining cause 仍然 unidentified。

Matched 11-paper benchmark 得到 Human = 387.5、Structured = 851.4、Free = 1874.4 mean words；Structured 约为 Human 的2.2倍，Free约为4.8倍。结合 RQ3 common coding，可将 Structured 的优势表述为 **relative concision without a clear recorded-coverage loss in the audited sample**。

## 4.5 Results summary

不再制作 cross-RQ summary table。四个 RQ 的 exact evidence 已由5张 figures与4张主表承载；本节末尾用一个 synthesis paragraph 强化记忆点：Structured 提供 targeted attenuation、relative concision 和 operational control，而 reliable logic verification 需要 complementary safeguards。

---

# 5. Discussion and Future Work

**章节职责：** 不重复 Chapter 4 的分析和数字。本章说明研究结论相对于 prior work 的新增认识，将 structure 定位为 targeted mitigation/operational control layer，提出 layered-system implications，并以紧凑 scope boundaries 导出 future work。

## 5.1 Structured reviewing as targeted mitigation and operational control

- 先概括 Structured setup 的成立价值：selected injection attenuation、machine-readability、relative concision 与 comparable common-coded coverage。
- Human benchmark 将 efficiency finding 与实际 reviewing practice 连接起来。
- Rating dispersion 作为独立伴随属性，不让其承担 Structured 价值的主要证明责任。

## 5.2 Relationship to manipulation-risk research

- 与 Ye et al. 的 white-text manipulation finding 对话：本研究确认 vulnerability，并把问题推进到 Free-versus-Structured mitigation comparison。
- 强调本文新增的 multi-outcome attenuation profile、common-coding corroboration 和 criticism-retention case evidence。
- 结论定位：structure 是 layered defence 中可量化的 behavioural constraint，而非孤立的 complete defence claim。

## 5.3 Relationship to faulty-reasoning evaluation

- 与 Dycke and Gurevych 的 counterfactual null finding 对话：本研究在 Free/Structured 两种 setups 中复现无 reliable logic-specific response。
- 本文的新增贡献是 target-level mechanism decomposition：target omission 为首要 observable failure，critique–rating decoupling 为 detected cases 的次级机制。
- 推出 evaluation implication：coverage fields 与 defect verification 是不同能力，未来应使用 target-aware criteria。

## 5.4 Implications for LLM-assisted peer-review design

- 提出三层 architecture：input-integrity checks、structured review layer、reasoning-verification layer。
- 讨论 Human-in-the-loop role：用 structured fields 组织 evidence，由 human reviewer 处理高风险 scientific claims 与 final judgement。
- 评价 ARG 时联合考虑 manipulation、targeted logic、coverage calibration 与 operational cost。

## 5.5 Scope and limitations

- 四组紧凑 scope boundaries：model/sample、compound setup、measurement/annotation、perturbation/deployment。
- Measurement triangulation 作为已实施的 design strength 先说明，再界定 single-annotator 与跨模型推广范围。
- 每个 limitation 只说明它限制哪类推广，不重新否定 within-paper findings。

## 5.6 Future work

- Guidance × format 的 2 × 2 factorial design，分离 Structured setup 的 active components。
- Target-aware reasoning verification，分别测量 detection、interpretation 和 rating consequence。
- Text-layer sanitisation、rendered-image comparison 与 multimodal defence evaluation。
- Multi-model/multi-annotator replication、repeated generations 和 human-in-the-loop usefulness study。

## 5.7 Discussion synthesis

- 结尾强化一个记忆点：Structured reviewing 是 practical control layer；它已经带来 measurable attenuation 与 concision gains，而 input integrity 和 logic verification 是其自然的 complementary modules。
- 过渡到 Chapter 6，不在最后重新展开 limitations。

---

# 6. Conclusion

建议的结论方向：

> 本研究在统一的 within-paper design 中，对比了 free-form 与 schema-constrained LLM reviewing setups 在 document-layer prompt injection、counterfactual logic defects、review-aspect coverage 和 operational efficiency 方面的表现。Structured setup 显著减弱了 rating inflation、injection compliance 与 methodological-criticism reduction，并大幅减少 review length、latency 和 token use；common coding 与 human benchmark进一步显示，这种 concision 在 audited sample 中没有伴随 clear recorded-coverage loss。与此同时，target-defect audit 说明 reliable logic verification 仍需要专门支持。由此，本文将 structured reviewing 定位为值得采用的 practical control layer，并提出以 input-integrity checks、reasoning verification 和 human oversight 构成 layered review system 的后续方向。

---

# Prioritised remaining work

## P0 — Chapter 4 正文

1. **Chapter 4 prose。** 按 RQ1 attenuation → RQ2 defect omission/decoupling → RQ3 measurement-sensitive coverage/granularity → RQ4 human-relative concision 的逻辑链起草正文。
2. **Chapter 4 synthesis。** 用一个 paragraph 收束，不重复四个 tables 的数字。

## P1 — 其他章节正文

1. 完成 Introduction 与 Related Work：research gap、Structured generation/review forms、ARG evaluation、contributions 和两篇关键文献。
2. 完成 Methodology 正文：sample description、pipeline diagram、metric definitions、triangulation 与 statistical procedures。
3. 按新细纲完成 Discussion：prior-work dialogue、layered-system implications、四组 scope boundaries 和四条 future-work routes。
