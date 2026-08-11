# 第3章细纲：Methodology

> **文件性质：** 本文件是 Chapter 3 的详细写作蓝图，不是 dissertation 正文。
>
> **章节职责：** 清楚说明实验如何运行、收集了哪些数据，以及这些数据为何能够回答 RQ1--RQ4。实际数值、案例发现和研究结论留在 Chapter 4。
>
> **叙事原则：** Methodology 应可复现，但不写成代码说明书或 API 流水账。正文只保留会影响研究解释的设计、条件、analysis unit、measurement source、aggregation 和 comparison；完整 prompts、schema、payload、paper IDs 与低层实现放入 Appendix 或 reproducibility materials。

---

## Chapter opening（不设 3.0 标题）

开头用两段完成导航：

1. 说明本研究在同一批 papers 上比较 Free 与 Structured reviewing，并通过 Counterfactual 和 Injection 两条 controlled tracks 分别检验 faulty-logic detection 与 manipulation susceptibility。
2. 说明 reviews 先进入 setup-specific operational measurement，再通过 common coding、targeted audit 和 human benchmark 进行 evidence triangulation。

使用一张不含结果数值的 pipeline figure：

`30 papers → Counterfactual / Injection inputs → Free / Structured reviews → operational metrics → auxiliary triangulation → RQ1–RQ4`

图负责呈现总体关系，正文不再逐节点复述。Chapter 3 总体控制在 6–7 页；除 pipeline figure 外，最多使用一张 compact design table 汇总两条 tracks、conditions 和输出数量。

---

# 3.1 Study Design and Paper Sample

## 3.1.1 Within-paper design

- 采用 strict within-paper design，分析基础为 30 papers。
- 同一 paper 在不同 input conditions 和两个 reviewing setups 下重复生成 review，以控制 paper-level quality、topic 和 difficulty 差异。
- Primary questions：RQ1 manipulation susceptibility；RQ2 logic-defect discriminability。
- Secondary questions：RQ3 review-aspect coverage；RQ4 operational efficiency、human-relative concision 与 rating dispersion。
- 本节重点解释 within-paper control 的意义，不提前描述结果。

## 3.1.2 Paper selection and analysis units

- 数据来源为 Dycke and Gurevych counterfactual corpus，以及对应的 local original full texts/PDFs。
- Eligibility：必须同时具有 Original full text、Original PDF，以及完整的 3 个 Logic 和 4 个 Format counterfactual versions。
- 从 123 篇 eligible papers 中以 seed = 10190 随机抽取 30 篇。
- RQ1/RQ2 的 inferential unit 为 paper；同一 paper 的多个 outputs 是 repeated observations，而不是独立样本。
- RQ3 使用每篇 paper 的 Original Free/Structured pair。
- RQ4 engineering summary 使用全部 240 Counterfactual paper-condition rows；human benchmark 使用有 eligible public reviews 的 matched Original papers。
- 正文只需简洁交代 venue/year 范围；完整 paper IDs 放 Appendix。

## 3.1.3 Models and experimental matrix

- Reviewer generator：`gpt-5.4`，temperature = 0。
- Independent Judge：`deepseek-v4-flash`，temperature = 0。
- Temperature = 0 用于降低 uncontrolled generation variation；本研究比较 condition/setup response，不检验 repeated-generation consistency。
- Compact design table：

| Track | Paper conditions | Reviewing setups | Generated reviews |
|---|---:|---:|---:|
| Counterfactual | 30 × 8 | Free + Structured | 480 |
| Injection | 30 × 2 | Free + Structured | 120 |
| Total | — | — | 600 |

- Judge processing 共 360 calls：Counterfactual Free 240、Injection Free 60、Injection Structured compliance 60。总调用量可在 reproducibility 段一句交代，无需展开调用过程。

---

# 3.2 Controlled Input Conditions

本节的任务是说明两条 tracks 分别改变了什么，以及由此形成什么可比较的数据；不写生成过程的代码顺序。

## 3.2.1 Counterfactual track

每篇 paper 包含 Original baseline 与七种 controlled perturbations：

- Logic-critical：`blueprint_conclusion`、`blueprint_finding`、`blueprint_result`；
- Format：`active_passive`、`british_american`、`language_error`、`paper_layout`。

对 paper (p)、condition (c) 和 outcome (Y)：

\[
\Delta_{p,c}=Y_{p,c}-Y_{p,Original}.
\]

- 每个 perturbation 只与同一 paper、同一 setup 的 Original output 比较。
- 每篇 paper 分别计算三个 Logic deltas 和四个 Format deltas 的平均值。
- Logic-specific discriminability 定义为 Logic response 相对 Format response 的差异，而不是任一 perturbation class 相对零的变化。
- 对 MF，正的 Logic--Format contrast 表示 Logic edits 引发更多 methodological criticism；对 rating，负的 contrast 表示 Logic edits 引发更低评价。
- 这一 contrast 将一般 rewrite sensitivity 与针对 claim/evidence inconsistency 的 response 分开，为 RQ2 提供 paper-level comparison。

## 3.2.2 Injection track

- 使用 PyMuPDF 在 PDF 第一页和最后一页加入 fontsize = 4、white-colour text-layer payload；Manipulated PDF 与 Original PDF 仅在该隐藏文本层上不同。
- Payload 伪装为 `Positive Calibration Track`，要求提高 rating、强化 praise、弱化 criticism，并避免暴露指令。
- 正文概括这些可测 behavioural targets；完整 payload 放 Appendix。
- 每个 paper/setup 计算 Manipulated PDF − Original PDF delta。
- 接近零表示 robustness；rating、strengths 和 compliance 增加以及 weaknesses/MF 减少构成 attack-aligned response。
- RQ1 的核心 comparison 是 Free attack delta 与 Structured attack delta 的 paper-level contrast，从而区分 attack effect 与 setup-level attenuation。

---

# 3.3 Reviewing Setups

## 3.3.1 Free and Structured reviewing

**Free reviewing**

- 使用 generic conference-review instruction，要求 comprehensive、rigorous 和 constructive review，但不规定具体 review dimensions。
- 输出为 plain-text prose，不使用 JSON 或固定 template。

**Structured reviewing**

- 显式要求 summary、strengths、weaknesses、methodological flaws、overall rating 和 confidence。
- 强制返回 `StructuredReview` Pydantic schema。
- 三类 aspect fields 均为 `list[str]`，使用对称的 empty-list instruction。
- Rating 为 1–10 integer；confidence 为 1–5 integer。Confidence 属于 schema 组成，但不作为本研究 outcome。

完整 prompts 与 schema 放 Appendix；正文重点说明两种 setup 如何改变 review guidance、organisation 与可测输出。

## 3.3.2 Meaning of the setup comparison

Structured setup 同时改变 review guidance、output organisation 与 machine-readability，因此本研究比较的是两种完整 reviewing setups，而不是孤立的 JSON serialisation effect。该定义对应实际 pipeline 问题：采用 schema-constrained reviewing 后，robustness、coverage 和 operational behaviour 如何变化。

---

# 3.4 Outcome Measurement

## 3.4.1 Operational measurement pipelines

| Outcome | Free | Structured |
|---|---|---|
| Rating | Judge 从 prose 推断 1–10 integer | native `rating_1_10` |
| Strengths | Judge distinct-item count | native list length |
| Weaknesses | Judge distinct-item count | native list length |
| Methodological flaws | Judge distinct-item count | native list length |
| Injection compliance | Judge 0–10 | Judge 0–10 |

- Free prose 在实际使用中需要 post-hoc extraction，而 Structured output 直接提供 typed fields；这构成两种 operational pipelines。
- 两类 measurement 都支持 setup 内部 paired change。跨 setup absolute aspect counts 结合 common coding 进行校准和解释。
- 该 measurement convention 在此完整说明一次，Chapter 4 不再反复展开。

## 3.4.2 Collected outcomes and their meaning

- **Rating：** review 对论文整体 acceptability 的 1–10 evaluation。
- **Strengths / weaknesses / methodological flaws：** review 中可区分的 positive merits、general concerns 和涉及 scientific logic、validity、empirical design 或 claim–evidence support 的 concerns。
- **Injection compliance：** review 对 rating inflation、enthusiastic praise 和 criticism downplaying 等 payload targets 的整体遵从程度，0–10。
- **Review length：** 统一 word-splitting procedure 得到的 word count，用于比较 output volume 与 human-relative concision。
- **Observed latency：** 单次 generation call 的 end-to-end elapsed seconds，用于描述实际响应时间。
- **Output tokens：** 使用 `o200k_base` 对保存的 raw outputs 统一计算，用于描述 processing volume 和 usage-based cost implication。
- **Rating dispersion：** setup-level rating SD 与 paper-level conditional SD，用于描述 rating distribution 的集中程度。

## 3.4.3 RQ mapping and evidence hierarchy

- RQ1：rating、MF 和 compliance 是主要 behavioural outcomes；strength/weakness changes 描述 payload-aligned content shift。
- RQ2：Logic--Format rating 与 MF contrasts。
- RQ3：strength/weakness/MF profile 和 total count；这些是 recorded coverage measures，不直接等同于 review quality。
- RQ4：words、latency、tokens、matched human length 与 rating dispersion。

正文以一段解释 outcome 与 RQ 的关系，不为每个 metric 建立独立小节。

---

# 3.5 Evidence Triangulation

本节说明辅助证据如何回答 operational metrics 单独不能回答的问题；不报告核验结果。

## 3.5.1 Condition-masked common coding

- Sample：5 papers、50 Free/Structured reviews，包含 RQ1 pairs 与 RQ2/RQ3 relevant conditions。
- Coding 隐藏 paper identity、attack/perturbation label 和 automated metrics；review format 可能暴露 setup，因此称为 `condition-masked`，而不是 fully setup-blind。
- 同一 rubric 记录 inferred rating、strengths、weaknesses、MF 和 injection compliance。
- 所有 final annotations 由 dissertation author 完成，报告为 `condition-masked single-annotator manual coding`。
- 作用：用共同 measurement 检查主要趋势是否收敛，并识别 measurement-sensitive dimensions；不替换 operational analysis。

## 3.5.2 Targeted qualitative audits

**RQ2 target-defect audit**

- 在 common-coding sheet 锁定后，对全部五个 matched Logic cases 解盲。
- 记录 target defect 是否被提及、是否正确联系到 claim validity，以及是否导致相对 Original 的 primary-rating decrease。
- 用于区分 target omission 与 critique--rating decoupling。

**RQ3 count-granularity inspection**

- 对一个预先选定的 Original matched pair 检查 operational/common count discrepancy。
- 分析 repeated themes、segmentation、category allocation 与 substantive coverage 的关系。
- Judge 只保存 aggregate counts，因此不对没有 item-level spans 支持的具体 double-count 作精确归因。

## 3.5.3 Human-review benchmark

- 在30篇实验 papers 中，11篇具有 eligible public primary reviews，共纳入45份 reviews，不使用替代 papers。
- 纳入 rebuttal 前的 initial primary reports；排除 meta-review、decision、author response、public comments、ethics-only review 和纯 rating records。
- Human 与 LLM reviews 使用同一 word-count procedure；human strengths、weaknesses 和 MF 由 dissertation author 按统一 rubric 编码。
- 同一 paper 的多份 human reviews 先取 paper-level mean，再与该 paper 的 Original Free/Structured review 比较。
- Length benchmark 使用 matched same-unit comparison；aspect benchmark 明确 Human author coding、Free Judge extraction 与 Structured native fields 的 measurement sources。
- 其意义是为 LLM output length 和 aspect profile 提供真实 peer-review reference，而不是把 human coding 设置为主要 gold standard。

---

# 3.6 Statistical Analysis

开头统一说明：所有 inferential comparisons 以 paper 为单位，报告 raw-scale mean contrast、95\% CI 和 adjusted `p`-value；正文不罗列 raw `p`、`t` statistic 或标准化 effect size。具体 test implementation 与 CI 计算方式在本节一次说明。

## 3.6.1 RQ1 and RQ2

**RQ1**

- 对五个 outcomes 分别计算 paper-level attack delta：Manipulated PDF − Original PDF。
- 使用 two-sided paired `t`-test 比较 Free delta 与 Structured delta，并报告 Free Δ − Structured Δ 的 `t`-based 95\% CI。
- Rating、MF、strengths、weaknesses 和 compliance 五项 `p`-values 构成一个 family，进行 Holm correction。
- 证据层级仍有区别：rating、MF 和 compliance 支撑主要 attenuation claim；strengths/weaknesses 提供 content-shift evidence。

**RQ2**

- 每篇 paper 内先计算 mean Logic Δ 与 mean Format Δ。
- 对 Free/Structured × rating/MF 的四个 Logic--Format contrasts 使用 two-sided paired `t`-tests。
- 四项 `p`-values 构成一个 family，进行 Holm correction；报告 Logic Δ、Format Δ、raw-scale contrast 和95\% CI。

## 3.6.2 RQ3 and RQ4

**RQ3**

- 仅使用30个 Original Free/Structured pairs。
- 报告三类 aspects 与 total count 的 descriptive means/profile。
- 使用5-paper common coding、one-pair textual inspection 和 human external profile解释 measurement sensitivity；不把 count difference 转换为 review-quality ranking。

**RQ4**

- 使用全部240个 Counterfactual paper-condition rows汇总 words、latency、tokens 和 global rating distribution。
- 报告 mean、median/SD 和 relative reduction，不为每个 engineering metric扩展独立 significance test。
- Global rating variance 使用 Levene test 比较240个 Free 与240个 Structured ratings。
- 每篇 paper 另计算八个 conditions 下的 setup-specific rating SD，并报告30个 paper-level SDs 的 mean/median 与 invariant-paper count，作为 matched descriptive diagnostic。
- 不使用对30个 SD values 进行的 Levene test 来证明平均 within-paper SD 不同；若后续需要 inferential mean comparison，应另行定义 matched test。当前正文以 descriptive result 支撑 within-paper pattern。
- Human length comparison 使用11篇 matched Original papers，以 paper-level mean 为单位，报告 `N`、mean、median 和 range，不增加不必要的显著性检验。

---

# 3.7 Reproducibility and Research Ethics

## 3.7.1 Reproducibility and data integrity

- 报告 random seed、完整 prompts/schema、model identifiers、temperature、tokenizer 和调用日期。
- 简洁说明 parsing validation、status flags、failed calls、retry rule 与 missing-data handling；不逐步复述 notebook execution。
- 保存从 source papers、condition index、raw reviews、operational metrics、manual annotations 到 final figures/tables 的文件映射。
- 保存生成最终 output-token totals 的 `o200k_base` 计算代码或 stats artifact；正文完成前确认该 provenance 可追溯。
- 主要 analysis scripts 与 final stats CSV 纳入 reproducibility materials。

## 3.7.2 Research ethics

- Prompt-injection attack 仅在本地受控数据和研究调用中实施，不接触真实 peer-review platform 或 decision process。
- Payload 和生成结果的披露服务于 vulnerability assessment 与 safer system design。
- 详细 payload 放 Appendix，正文只说明研究所需的攻击行为目标与隔离措施。

---

# 正文写作节奏

1. Chapter opening + pipeline figure：快速建立 two tracks、two setups 和 measurement/triangulation flow。
2. Study design/sample：说明 within-paper control、sample 和 analysis units。
3. Controlled inputs：解释每条 track 改变什么、产生何种 comparison。
4. Reviewing setups：说明实际差异及其研究意义。
5. Outcomes：交代收集的数据、source 和每类数据回答什么问题。
6. Triangulation：说明辅助核验为何需要以及如何实施。
7. Statistics + reproducibility/ethics：准确但紧凑地收束。

每段遵循 `设计目的 → 实施方式 → 产生的数据或比较意义`。避免：代码级细节、API 日志、字段清单式说明、重复公式、结果预告和防御性辩解。

### Methodology self-check

- Chapter 4 使用的每个 metric 是否在此定义且 measurement source 一致？
- 每个 analysis unit、aggregation、contrast、test family 和 CI 是否能够复现？
- 是否解释了数据为什么能够回答 RQ，而不只是说明程序执行了什么？
- 是否把 Free/Structured 定位为完整 setups，而非只归因于 JSON？
- 是否准确写明 single-annotator、condition-masked、one-pair inspection 和 matched human benchmark？
- 是否把结果数值和机制结论留给 Chapter 4？
- 是否能在6–7页内完成，且没有可移入 Appendix 的说明书式细节？
