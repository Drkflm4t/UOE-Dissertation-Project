# Dissertation Writing Standard

## 1. 总体目标

将论文写成一条清晰、可信且有记忆点的学术论证，而不是项目总结、实验日志或免责声明。正文应主动告诉读者：研究解决了什么问题、发现了什么、证据为何足以支持该结论，以及这些发现为什么重要。

语言保持正式、简洁和易读，约为 IELTS 7.5 的学术表达水平。优先使用准确的普通词汇和直接句式；专业术语只在必要时使用，并在首次出现时定义。

## 2. 全文通用原则

细纲负责约束证据边界，正文负责清楚、有力度地呈现发现。不能把细纲中的Claim boundary转写成免责声明。
1. **发现优先。** 段落先陈述结果或论点，再给数据、解释和必要边界；不要用限制开场。
2. **主张与证据对齐。** 明确说出证据支持的结论，不夸大到设计未测量的概念，也不把已有证据写成“什么都不能说明”。
3. **一段一个任务。** 首句说明本段信息，后续句子依次提供证据、解释或过渡。不能映射到章节主线的段落应删除或合并。
4. **细纲不等于正文标题。** 细纲中的编号是论证检查点，不应逐项转化为 `section/subsection`。正文只在研究问题、方法模块或证据阶段发生实质转折时设置标题；判据、单项结果、案例和小结优先通过 topic sentence 串成连续论证。
5. **以最终逻辑组织，而非研究过程。** 不按“先尝试—失败—再修改”的时间顺序写作；只呈现理解当前设计和结论所需的最终方案。
6. **证据链胜过免责声明。** Quantitative results、common coding、human benchmark 和 qualitative cases 各自承担明确功能。多种证据汇合时，应直接说明它们共同支持什么。
7. **限制集中且不重复。** 只有会实质改变结论解释的边界才在 Results 就近说明一次；一般性的 scope、internal mechanism 和 future extension 集中放入 Chapter 5。
8. **诚实但不自我削弱。** 不隐藏 null results、trade-offs 或必要限制，但将其准确定位为研究发现、适用范围或后续问题，而不是对整项工作的笼统负面评价。
9. **术语和统计口径保持一致。** RQ、setup 名称、metric、analysis unit、measurement source 和显著性口径在全文统一；避免同一概念反复换名。
10. **图表各自只传达一个主要信息。** Caption 交代样本、口径和读图方式；正文解释图表意味着什么，不逐格复述数字。
11. **章节各司其职。** Methodology 说明如何得到证据；Results 完成主要数据分析与机制解释；Discussion 将发现置于既有研究和应用语境中，并集中处理 limitations 与 future work。

## 3. 证据强度与措辞

- 直接统计检验和多源证据一致：使用 `demonstrates`、`shows`、`provides strong/converging evidence`。
- 稳定的描述性模式或有限样本验证：使用 `indicates`、`supports the interpretation`。
- 探索性现象或单一案例：使用 `suggests`、`is consistent with`。
- 设计确实无法区分的解释：简洁写明 `cannot distinguish`，随后将问题移交 Chapter 5，不在各节反复提醒。

自信来自证据与措辞匹配，不来自夸张，也不来自回避不利结果。

## 4. 句段检查

每个段落完成后检查：

1. 首句是否直接表达本段结论或功能？
2. 每句话是否支持、解释或推进首句？
3. 是否重复了前文已经披露的限制？
4. 是否明确说出了数据指向什么？
5. 删除一句后若论证不受影响，该句是否应删除？

---

# 各章正文写作原则

以下各章均遵循同一结构转换规则：详细细纲用于检查逻辑和证据是否完整，最终正文则将相邻检查点合并为少量连贯的 narrative units。标题负责导航主要论证阶段，段落首句负责导航阶段内部的具体内容。

## 全文篇幅预算

`mscdiss-skeleton` 使用 A4、12pt、1.5 倍行距和固定页边距；不得通过缩小字体、间距或 margins 压缩篇幅。模板将 Chapter 1 记为正文第1页，并要求正文在第40页前结束；preliminary pages、bibliography 和 appendices 位于该正文预算之外。

本文将 **Chapters 1–6（包含正文中的 figures 和 tables）控制在29–33页，排版目标为约31–32页**。各章预算如下：

| Part | Target pages | Approximate share |
|---|---:|---:|
| Chapter 1 — Introduction | 3–3.5 | 11% |
| Chapter 2 — Related Work | 4–4.5 | 14% |
| Chapter 3 — Methodology | 6–7 | 21% |
| Chapter 4 — Results and Analysis | 11–12 | 37% |
| Chapter 5 — Discussion, Limitations and Future Work | 3.5–4.5 | 13% |
| Chapter 6 — Conclusion | 1–1.5 | 4% |
| **Main-body total** | **29–33** | **100%** |

页数是论证预算而不是必须填满的配额。写作期间应使用最终模板定期编译，以 Chapter 6 结束页为准；若超出预算，优先合并过密标题，并删除重复背景、重复数字、重复 caveats 和非必要图表，不改变模板版式。Abstract 建议控制在250–350 words，通常占一页 preliminary material，不计入上述29–33页。

## Abstract

**篇幅：250–350 words，约1页 preliminary material。** 按 `problem → gap → design/contribution → strongest findings → significance` 展开。只保留最重要的结果与贡献，不罗列全部指标，不在贡献尚未建立前讨论限制。摘要中的每项主张必须能由 Chapter 4 直接支持。

## Chapter 1 — Introduction

**篇幅：3–3.5页，约占正文11%。** 从 LLM peer review 的现实价值与 manipulation/reasoning risks 切入，迅速建立“structured reviewing 能否缓解这些缺陷”的研究缺口。清晰给出 RQs、研究设计概念和贡献；不要提前讲实验细节，也不要用大段 limitations 削弱研究动机。篇幅重点放在 problem、gap 和 contributions，背景综述移交 Chapter 2。

## Chapter 2 — Related Work

**篇幅：4–4.5页，约占正文15%。** 按研究主题而非年份罗列组织文献：automatic review generation、manipulation vulnerability、faulty-reasoning detection 和 structured generation/evaluation。每节完成 `已有工作做了什么 → 仍缺少什么 → 本文如何推进`，公平呈现最相关研究，不制造 strawman，也不写成 citation list。只保留能够建立 research gap 或解释设计选择的文献。

## Chapter 3 — Methodology

**篇幅：6–7页，约占正文21%，包含一张 pipeline/design visual，并按需要使用一张 compact design table。** 本章要让读者清楚理解实验如何运行、收集了哪些数据，以及这些数据为什么能够回答各 RQ。按 `study design and sample → controlled input conditions → reviewing setups → outcome measurement → evidence triangulation → statistical analysis → reproducibility and ethics` 描述最终 pipeline。Free-Judge 与 Structured-native 的 measurement procedure 在此完整披露一次，同时说明 common coding 如何校准并部分弥合 measurement gap。

Methodology 应保持可复现但不写成代码说明书：正文描述对研究结论有意义的流程、条件、analysis unit、metric source、aggregation 和 comparison，不逐项复述 notebook 顺序、API 调用日志、class 定义或每个字段的实现细节。完整 prompts、schema、payload、paper IDs 和低层实现移入 Appendix 或 reproducibility materials。每个方法段落按 `为什么需要该设计 → 如何实施 → 产生什么可分析数据` 展开；不提前报告结果，也不反复为设计辩护。

最终正文不复制详细细纲，使用以下七个主要部分：`Study Design and Paper Sample`、`Controlled Input Conditions`、`Reviewing Setups`、`Outcome Measurement`、`Evidence Triangulation`、`Statistical Analysis`、`Reproducibility and Research Ethics`。相邻的模型设置、analysis unit、metric hierarchy 和 RQ mapping 通过段落整合，不再各自设置短小标题。

## Chapter 4 — Results and Analysis

**篇幅：11–12页，约占正文37%，包含5张主图和4张主表。** 本章是全文篇幅最大、也是最主要的 empirical argument，而不是免责声明。每个 RQ 按 `quantitative finding → triangulation/case evidence → mechanism-level interpretation → direct RQ answer` 展开，先说发现，再用证据完成论证。图表预计占本章约3.5–4.5页，其余版面用于解释关键 effect、evidence convergence 和 RQ answer；不逐项复述表中数字，也不新增重复图表。

最终标题结构不复制详细细纲：chapter opening 不设 `4.0` 标题；每个 RQ 使用一个 `section`，并通常只保留两个承担主要论证转折的 `subsection`。判据放在 section opening，RQ answer 放在该 section 的结尾段，不另设标题。计划结构为：

- **RQ1：** `Quantitative Attack Effects`；`Methodological Criticism and Evidence Triangulation`。
- **RQ2：** `Aggregate Discriminability`；`Target-Defect Audit`。
- **RQ3：** `Operational Coverage Profile`；`Common Coding and Human Benchmark`。
- **RQ4：** `Efficiency and Human-Relative Concision`；`Rating Dispersion`。
- **Cross-RQ synthesis：** 只保留一个短 section 或收束段，不再展开子标题。

- 4.0 集中交代一次 Free-Judge 与 Structured-native 的 measurement setup。后文不重复“二者不可绝对比较”；RQ3 只需强调 common coding 如何主动校准比较并确认哪些趋势稳定。
- 将细纲中的 `Claim boundary` 视为内部写作护栏，不逐条转写进正文。每个 RQ 最多保留一句真正影响解释的边界，其余集中放入 Chapter 5。
- Human length、matched common-coded totals 与 qualitative audit 构成联合证据。应明确写出：Free 的额外长度主要反映更高 verbosity 和 segmentation，而不是可靠的 substantive-coverage gain；不要在结论后立即叠加多层自我否定。
- RQ2 直接报告 observable behaviour：主要失败是 target-defect omission；即使缺陷被批评，也很少转化为 rating decrease。无需反复讨论不可观测的 internal awareness。
- RQ4 直接报告 concision、latency、token use 和 rating dispersion。对无法识别的内部 compression mechanism 只作一次简短说明，不让其遮盖 operational contribution。
- 使用自信而严谨的语法：告诉读者“多源证据共同说明了什么”，而不是连续列举“数据不能证明什么”。

## Chapter 5 — Discussion, Limitations and Future Work

**篇幅：3.5–4.5页，约占正文13%。** 不重复 Chapter 4 的数字和逐项分析。围绕三件事展开：研究发现如何扩展或修正已有文献；structured reviewing 在 layered review system 中承担什么作用；哪些 scope boundaries 指向下一步研究。Limitations 只保留真正影响推广或因果解释的内容，并与具体 future work 一一对应，不写成对项目的重新审判。Discussion 的长度来自与 prior work 和 system implications 的深入对话，而不是重复 Results。

## Chapter 6 — Conclusion

**篇幅：1–1.5页，约占正文4%。** 强化一个最终记忆点：structured reviewing 是具有 measurable attenuation、concision 和 machine-readability 优势的 practical control layer，但可靠的 input integrity 与 logic verification 需要 complementary safeguards。概括最强证据和实际意义，不引入新分析，不在结尾突然扩大限制或否定贡献；若内容不足，不为达到页数而重复各 RQ。
