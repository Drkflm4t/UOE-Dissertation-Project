# Post-unblinding Qualitative Case Audit

## Scope and evidential status

This memo records the hypothesis-led checks conducted after the blinded annotation sheet was locked and unblinded. It does not modify the blind coding, does not estimate population prevalence, and is not a human gold-standard evaluation. The RQ2 audit uses all five Logic matched cases in the manual-validation sample; the RQ3 inspection uses one preselected Original matched pair.

## RQ2 — Target-defect audit

### Coding convention

- `target_defect_mentioned`: 1 only when the review refers to the altered result/claim or its resulting claim–evidence inconsistency.
- `target_defect_correctly_interpreted`: 0 = absent/incorrect; 1 = partial; 2 = correctly links the edit to claim validity.
- `rating_consequence_present`: 1 only when the target is detected **and** the setup's primary rating decreases relative to its matched Original review. Free uses Judge-extracted rating; Structured uses native rating. A more negative inferred tone without a primary-rating decrease is reported separately, not coded as a rating consequence.

### Case-level results

| Case | Target edit | Structured M/C/R | Free M/C/R | Interpretation |
|---|---|---:|---:|---|
| 1 — `2024.acl-long.375`, `blueprint_result` | FOLIO result changed from 0.849 to 0.789, weakening the claim of approaching GPT-4 (0.855); 0.789 still exceeds GPT-3.5's 0.767 | 1/2/0 | 1/2/0 | Both reviews challenged the overstated GPT-4-level claim; neither primary rating decreased. |
| 2 — `2024.emnlp-main.758`, `blueprint_conclusion` | Unsupported attention-based pruning and estimated 10% accuracy claim added | 0/0/0 | 1/2/1 | Free explicitly stated that Table 7 did not support the claim and its primary rating decreased by one point. |
| 3 — `2024.emnlp-main.1123`, `blueprint_result` | Results rewritten as underperformance while the same paragraph retained a contradictory positive result | 1/2/0 | 1/2/0 | Both reviews correctly identified the contradiction, but both primary ratings were unchanged; this is the clearest critique–rating decoupling case. |
| 4 — `NeurIPS 2cQ3lPhkeO`, `blueprint_conclusion` | Unsupported SFT-effectiveness conclusion inserted and duplicated in the research-question position | 0/0/0 | 0/0/0 | Free noticed duplicated prose at a surface level but did not identify the targeted unsupported scientific conclusion. |
| 5 — `2024.acl-long.741`, `blueprint_result` | Privacy-reduction score changed to 2.0 while an unmodified 3.2 remained elsewhere, creating an internal inconsistency | 0/0/0 | 0/0/0 | Neither review identified the 2.0–3.2 inconsistency. Free's reference to 3.2 was grounded in an unmodified occurrence and should not be described as hallucination. |

`M/C/R` denotes mentioned / correctly interpreted / rating consequence.

### Aggregate mechanism

- Structured detected 2/5 target defects; Free detected 3/5. With only five cases, this is not a reliable setup difference.
- Across the ten setup-specific reviews, 5/10 omitted the target and 5/10 detected it under the semantic coding rule above. A stricter rule requiring explicit quotation of the altered number would classify Case 1 Free as an omission, yielding 6/10 omissions; this sensitivity does not alter the substantive conclusion.
- Only 1/5 detected reviews produced a lower primary rating attributable to the target defect. The remaining detected reviews criticised the defect without lowering the primary rating.

The aggregate RQ2 null therefore reflects a **mixture of target omission and critique–rating decoupling**, rather than either mechanism alone. Schema constraint did not produce a reliable detection advantage.

### Representative dissertation example

Use Case 3. Both reviews explicitly identified the contradiction between the rewritten underperformance claim and the retained positive result, yet Free and Structured primary ratings both remained unchanged relative to Original. This single case cleanly illustrates critique–rating decoupling without implying that all target defects were detected.

## RQ3 — Count-granularity inspection

### Selected pair

`2024.emnlp-main.758`, Original: MV_034 (Free) versus MV_002 (Structured). The inspection focuses on the Free review because its automatic/common discrepancy is large.

| Free measure | Strengths | Weaknesses | MF | Full total |
|---|---:|---:|---:|---:|
| Automatic Judge | 10 | 14 | 11 | 35 |
| Common auxiliary coding | 7 | 3 | 14 | 24 |

The often-cited 25 versus 17 comparison is the **critical-aspect count** (`weaknesses + MF`), not the full aspect total. Its net gap is 8. The 11-item weakness discrepancy is partly offset by three additional common-coded MF items, showing category reassignment as well as a possible segmentation difference.

### What the text supports

- The long Free review repeatedly returns to broad themes such as comparison fairness, confounded components and insufficient experimental control.
- The automatic and common procedures allocate the same text very differently across weaknesses and MF.
- The discrepancy is consistent with finer automatic segmentation and sensitivity to the weakness/MF category boundary.

### What the text does not establish

- The Judge returned counts only, not item-level spans or lists. It is therefore impossible to reconstruct which exact sentences it counted separately or in both categories.
- Several apparently related concerns are independently resolvable—for example manual-shot scaffolding, uncontrolled larger-model comparisons, equation notation, dimension definitions and epoch inconsistency. Counting these separately is not automatically erroneous under the distinct-item rule.
- Pair A therefore does not prove that the Judge mechanically split or double-counted the specific examples proposed during inspection.

### RQ3 conclusion supported by the case

> The automatic and common coding procedures produced substantially different category allocations and critique counts for the same Free review. The pattern was consistent with segmentation and category-boundary sensitivity, but the count-only Judge output did not permit direct identification of item-level duplication or over-segmentation.

This case explains why the automatic Free > Structured ordering should be treated as measurement-sensitive. It does not show that Structured reviews are more comprehensive or higher quality.
