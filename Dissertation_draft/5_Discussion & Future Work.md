# 第5章细纲：Discussion and Future Work

> **文件性质：** 本文件是 Chapter 5 的详细细纲和写作蓝图，不是 dissertation 正文。
>
> **章节职责：** Discussion 不重新完成 RQ1–RQ4 的数据分析，也不重复主要数字。本章从 Chapter 4 已建立的结论出发，说明本研究相对于 prior work 增加了什么认识、structured reviewing 适合承担什么角色、结论适用于什么范围，以及哪些后续研究最值得推进。
>
> **叙事原则：** 以研究贡献和可操作启示为中心，而不是以 limitations 清单为中心。必要的 scope boundaries 集中放在一节，并直接导向 future work；不使用自我贬低式语言，也不把未检验的问题主动扩展成新的研究责任。

---

## 5.0 本章主线

Chapter 4 已经建立四项互相衔接的发现：

1. Structured reviewing 显著减弱了部分 prompt-injection effects，但两种 setups 都会受到 document-layer instruction influence。
2. 两种 setups 都没有形成可靠的 logic-specific discriminability；target audit 将主要 observable mechanism 定位为 defect omission，并识别 critique–rating decoupling。
3. Free 的 higher operational aspect count 主要集中于 measurement-sensitive weaknesses；共同 coding 没有显示 Structured 的 total recorded coverage 明显下降。
4. Structured 显著减少 length、latency 和 tokens，并在 matched human benchmark 中更接近实际 human-review length。

Discussion 的核心观点为：

> Structured reviewing should be understood as a valuable control layer for organising outputs, reducing operational cost and attenuating selected manipulation effects. Its strongest use is therefore as part of a layered reviewing system, complemented by input-integrity checks and targeted support for scientific-reasoning verification.

这一定位同时说明本文的贡献和下一步方向，不把 structure 描述成万能方案，也不把其价值因未解决全部问题而削弱。

---

# 5.1 Structured reviewing as targeted mitigation and operational control

## 段落1：最重要的综合发现

开篇直接说明本文带来的新认识：review structure 不只是 formatting choice，它会系统性改变 manipulation response、criticism retention、output length、machine-readability 和 rating distribution。Structured setup 的价值最清楚地体现在两个方面：

- selected injection effects 的幅度较小；
- review generation 更 concise、faster、less token-intensive，同时 common coding 未显示 clear total recorded-coverage loss。

本段不重复全部数值，只选择最有记忆点的结论语言：`partial mitigation`、`relative concision`、`comparable common-coded coverage`。

## 段落2：为什么这是有意义的贡献

- 现有研究往往直接评估某个 ARG 或某种攻击；本文比较的是可部署的 Free/Structured reviewing setups。
- 结果表明输出组织可以成为实际 system-design variable：它不能替代安全/推理模块，但能改变攻击影响幅度和资源成本。
- Human benchmark 强化了 practical relevance：Structured 相比 Free 更接近 human reviewing 的篇幅尺度，说明效率差异不仅是两个 prompts 之间的相对差异，也与实际 review practice 有联系。

## 段落3：rating dispersion 的位置

- Structured rating distribution 更窄是稳定观察到的伴随属性。
- 当前数据支持描述 `lower dispersion`，但不需要让这一现象承担 Structured 价值的主要证明责任。
- 一致性、scale compression 与 sensitivity 应由专门 repeated-generation/human-calibration design 区分；本研究的强贡献仍是 robustness attenuation、coverage triangulation 与 operational concision。

---

# 5.2 Relationship to manipulation-risk research

## 与 Ye et al. 的一致性

Ye et al., *Are We There Yet? Revealing the Risks of Utilizing Large Language Models in Scholarly Peer Review*，证明 small white-text instructions 可以操纵 LLM-generated reviews，引发 rating inflation、positive-review shift 和与 human reviews 的 alignment reduction。

本研究与其核心结论一致：document-layer hidden instructions 能够改变自动 review behaviour，说明 manipulation vulnerability 不是单一 pipeline 的偶发现象。

## 本研究的扩展

本文在三个方面推进该问题：

1. **Structured-versus-Free comparison：** 不只证明攻击有效，还检验 review structure 是否改变攻击幅度。
2. **Multi-outcome analysis：** 同时追踪 rating、strengths、weaknesses、methodological criticism 与 compliance，揭示 attenuation 在不同 behavioural dimensions 上具有明确 profile。
3. **Text-level mechanism evidence：** common coding 与 matched case 显示 Structured mandatory fields 在受到攻击时仍可保留 substantive criticism，为 observed attenuation 提供 output-level explanation。

讨论的重点应是：本文将 prior work 的 vulnerability finding 转化为一个 mitigation question，并证明 structured setup 能够提供 measurable but incomplete attenuation。因此更合适的设计方向是把 structure 作为 layered defence 中的 behavioural constraint，而不是把“是否完全阻止攻击”作为唯一价值标准。

---

# 5.3 Relationship to faulty-reasoning evaluation

## 与 Dycke and Gurevych 的一致性

Dycke and Gurevych, *Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework*，通过 controlled counterfactual edits 发现 faulty research logic 对 ARG reviews 没有显著影响。

本研究复现了这一核心 pattern：无论 Free 还是 Structured，Logic perturbations 相对 Format perturbations 都没有产生可靠的 rating 或 MF-specific response。

## 本研究的扩展

本文增加了两层 prior framework 中没有直接回答的证据：

1. **Review-structure comparison：** 显式 methodological-flaw fields 并没有自动转化为 logic-specific discriminability，说明 coverage prompts 与 defect verification 是不同能力。
2. **Target-level audit：** aggregate null result 被进一步拆解为两个 observable stages：首先是 target-defect omission；在部分已识别 cases 中，又出现 critique–rating decoupling。

这一机制分解是重要贡献，因为它将“review 没有变化”转化成更具体的 system requirements：未来系统既需要提高 target retrieval/verification，也需要把局部 critique 稳定整合进 overall evaluation。

## Literature-facing interpretation

- Free reviews 可以生成更多批评，但更多 criticism 不等同于对 manipulated defect 的 specificity。
- Structured fields 可以保证某类内容有固定位置，却不能保证其中的 criticism 指向正确的 claim–evidence inconsistency。
- 因此，未来 ARG evaluation 应继续采用 targeted counterfactuals 和 target-level detection criteria，而不是仅使用 total critique counts 或总体 review similarity。

---

# 5.4 Implications for LLM-assisted peer-review design

## 5.4.1 A layered system architecture

从本文结果可导出三个互补层次：

1. **Input-integrity layer：** 在 review generation 前检查 PDF text layer、rendered pixels 与 OCR/extracted text 的不一致，识别 invisible/off-page/small-font instructions。
2. **Structured review layer：** 使用 schema-constrained fields 组织 summary、strengths、weaknesses、MF 和 rating，获得 machine-readability、relative concision 和 selected behavioural attenuation。
3. **Reasoning-verification layer：** 对关键 results、conclusions 和 findings 建立 target checks，要求 review 显式验证 claim–evidence relations，并将 detected defects连接到评分理由。

该 architecture 是本文四个 RQ 的统一设计启示：structure 保留其最强优势，同时由专门模块处理它没有被设计来独立解决的问题。

## 5.4.2 Human role

- 当前证据更支持将 LLM reviews 用作 structured decision support，而不是单独作出 final review decision。
- Human reviewers 可重点检查高风险 scientific claims、跨段落 consistency 和模型标记的 methodological issues。
- Structured output 的 machine-readable fields 可以帮助人类快速定位 evidence，而 human benchmark 提醒系统设计避免用不必要的篇幅增加审核负担。

## 5.4.3 Evaluation practice

- Robust ARG evaluation 应同时包括 adversarial input、targeted logic counterfactuals、coverage/measurement calibration 和 operational cost。
- 单一 overall quality score 无法展示这些 trade-offs；本文的多层评价框架本身构成可复用的 evaluation perspective。

---

# 5.5 Scope and limitations

本节控制为四个紧凑段落。每段先说明结论适用范围，再给出最直接影响，避免形成十项自我批评清单。

## 5.5.1 Model and sampling scope

- 主要实验使用一个 Generator、一个 Judge 和30篇满足完整 counterfactual intersection 的 AI/NLP papers。
- 结论直接适用于当前 models、prompts 和 paper domain；跨模型、跨领域稳定性需要 replication。
- 这是 scope boundary，不改变 within-paper comparisons 对当前 setup effect 的识别价值。

## 5.5.2 Setup scope

- Free 与 Structured 是完整 pipeline-level setups；Structured 同时改变 dimension guidance、output organisation 与 extraction route。
- 因此本文识别的是采用 structured reviewing setup 的整体结果，而不是 JSON syntax 的独立 causal effect。
- 若需要分离各组成部分，应采用 guidance × format factorial design。

## 5.5.3 Measurement scope

- Primary operational metrics 保留 Free-Judge 与 Structured-native 的实际 pipeline；5-paper common coding 部分对齐 measurement rubric，Human profile提供 external reference。
- Auxiliary 和 Human annotations 均由 dissertation author 完成，因此不提供 inter-rater reliability。
- 该 design 足以支持 convergence/sensitivity analysis；更大规模 multi-annotator coding 可进一步提高 measurement generality。

## 5.5.4 Perturbation and deployment scope

- Counterfactuals 是 controlled edits，Injection track 检验一种 representative hidden-text payload。
- 每个 condition 单次 deterministic generation，因此本研究报告 rating dispersion，而不把它等同于 test–retest consistency。
- API latency 是 observed end-to-end latency，适用于当前 deployment conditions。

---

# 5.6 Future work

Future work 只保留与上述 scope 和主要发现直接相连的四条路线。

## 5.6.1 Isolate the active components of structure

实施 2 × 2 factorial design：

- generic guidance + prose；
- dimension guidance + prose；
- generic guidance + JSON；
- dimension guidance + JSON。

该设计将区分 review rubric guidance、serialization 和 machine-readable extraction 的独立贡献。

## 5.6.2 Strengthen reasoning verification

- 将 counterfactual target locations 或 research-logic graph 用于 target-aware review generation/evaluation。
- 分别测量 detection、correct interpretation 和 rating consequence。
- 研究显式 claim–evidence checklist、retrieval support 或 verifier module 是否减少 omission 和 critique–rating decoupling。

## 5.6.3 Evaluate layered manipulation defence

- 测试 text-layer sanitisation、rendered-image comparison、OCR discrepancy detection 和 multimodal reviewing。
- 扩展 payload wording、position、font/colour、off-page text 与 indirect instructions。
- 评价 defence 不仅是否拦截 payload，也评价对正常 paper content 和 review usefulness 的影响。

## 5.6.4 Extend reliability and human evaluation

- 在多个 generators、judges、domains 和 seeds 上 replication。
- 对每个 paper-condition 进行 repeated generations，直接测量 test–retest reliability/ICC。
- 使用 multi-annotator coding 验证 coverage、target detection 与 review usefulness。
- 将 structured reviews 嵌入 human-in-the-loop study，测量其是否减少审阅时间并提高问题定位效率。

---

# 5.7 Discussion synthesis

结尾只保留一个记忆点，不重复 limitations：

> The findings position structured reviewing as a practical control layer rather than a complete reviewing solution. It provides measurable attenuation of selected manipulation effects and substantial gains in concision and machine-readability, while the target-defect audit shows where dedicated integrity checks and reasoning verification add complementary value.

该段自然过渡到 Chapter 6：本文不仅确认了两个已知风险，还识别了 structure 能够实际改善的维度，并给出更合理的 layered-system direction。

---

# 正文写作角色与篇幅建议

1. **Opening synthesis（约10%）：** 不带数字地概括结构化 setup 的价值定位。
2. **Prior-work dialogue（约30%）：** 分别讨论 manipulation 与 faulty reasoning 两篇关键文献，以及本文的扩展。
3. **System implications（约25%）：** layered architecture、human role 和 evaluation practice。
4. **Scope and limitations（约20%）：** 四个紧凑范围段落。
5. **Future work + synthesis（约15%）：** 四条直接、可执行路线和结尾记忆点。

### Discussion self-check

- 是否避免重复 Chapter 4 的完整数字和案例分析？
- 是否明确说出本文相对两篇 key references 的新增贡献？
- 是否把 structure 的优势放在本章中心，而非以 limitation 开篇？
- 每项 limitation 是否只是界定 scope，并对应一项 future direction？
- 是否避免把未经验证的 cognitive mechanism 写成事实？

### Claim–evidence map

- Claim：Structured reviewing 是有价值的 targeted mitigation/control layer。| Evidence：RQ1 attenuation + RQ4 efficiency/concision。| Status：supported。
- Claim：Structured brevity 没有在 audited sample 中表现出 clear recorded-coverage loss。| Evidence：5-paper common totals + human length benchmark。| Status：supported within audited scope。
- Claim：Output structure alone does not produce logic-specific defect detection。| Evidence：RQ2 contrasts + target audit。| Status：supported for current setup。
- Claim：Layered input-integrity and reasoning-verification modules are promising design directions。| Evidence：由 observed failure stages 导出的 system implication。| Status：recommendation, requires future evaluation。
