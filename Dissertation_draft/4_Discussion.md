# Discussion: Analysis Outline

## RQ1: Manipulation Robustness — White-Text PDF Injection

### Key Results
- **Free (Judge):** ATE = +1.90 (p < 0.001) — injection inflated ratings by nearly 2 points
- **Struct (Self):** ATE = +0.97 (p < 0.001) — Pydantic integer firewall compressed the effect by ~49%
- **Gap:** 0.93 points — structured output partially dampens but does not eliminate adversarial influence

### The Firewall Paradox
- Struct's `rating_1_10` appears to "resist" injection (ATE +0.97 vs +1.90), but other dimensions tell a different story:
  - `n_strengths` Δ = +2.10 (p < 0.001) — LLM enthusiastically listed more strengths
  - `n_weaknesses` Δ = −1.20 (p < 0.001) — LLM suppressed weakness reporting
  - `n_methodological_flaws` Δ = −1.83 (p < 0.001) — LLM "forgave" genuine flaws
- **Interpretation:** Pydantic constrains the single integer `rating_1_10`, but the textual content (strengths/weaknesses/MFs) embedded within the same JSON is fully contaminated by the injection. Structured output creates a *false sense of security* — the rating number looks defended, but the underlying review text is not.

### Slope Graph (Figure 1)
- All 30 papers' Free ratings shift upward after injection (crimson lines dominate)
- Struct shows fewer crimson lines — partial dampening visible visually

### Bubble Chart (Figure 2)
- Free reviews cluster in upper-left quadrant: Score↑ + Weaknesses↓ = full behavioral compliance
- Bubble size (Δ Strengths) amplifies the effect — enthusiastic praise correlates with higher scores

### Scatter — Scientific Integrity (Figure 3)
- Free: Strong negative correlation between Δ Score and Δ MFs — injection causes "MF blindness"
- Struct: Weaker correlation — the Pydantic firewall partially obscures the relationship

### Discussion Points
1. Why does the injection succeed despite the "system calibration" disguise? (Psychological framing avoids safety guardrails)
2. Implications for real-world peer review systems using LLMs with PDF ingestion
3. Vision API as natural defense: PDF→image rendering strips text-layer injections

---

## RQ2: Defect Discriminability — Logic vs. Surface Perturbations

### Key Results
- **Methodological Flaws:** Neither Free (Judge) nor Struct (Self) detected significant Δ between Logic-Perturbed and Format-Perturbed conditions
- **Score:** Similarly, no condition produced consistent score degradation
- All boxplots cross zero for both Logic and Format perturbation types

### Interpretation
- LLM reviewers fail to discriminate between genuine logical defects (blueprint perturbations) and superficial format changes (passive voice, British/American spelling, language errors, layout)
- This is true for both Free (natural text) and Struct (schema-constrained) paradigms
- **Implication:** LLM-based review is not a reliable quality gate for scientific manuscripts — it treats deep logical errors and surface typos with equal indifference

### Discussion Points
1. Why do Logic-Perturbed papers fail to trigger higher MF counts? (LLMs evaluate surface fluency, not logical validity)
2. Comparison with human reviewers' known sensitivity to logical flaws
3. The "librarian, not a scientist" critique: LLMs catalog features but don't reason about correctness

---

## RQ3: Review Comprehensiveness — Struct Self vs. Free Judge

### Key Results
- Struct Self (Original condition): 22.3 items/review (strengths + weaknesses + MFs)
- Free Judge (Original condition): 26.2 items/review
- Struct produces ~15% fewer distinct critique items than Free

### Interpretation
- The Pydantic schema imposes a "comprehensiveness ceiling" — list fields create an implicit length constraint
- However, this may be a feature rather than a bug: Struct's items are more structured and machine-readable
- The trade-off is between *breadth* (Free generates more distinct points) and *extractability* (Struct's points are already parsed)

### Discussion Points
1. Is "more items" always better? (Signal-to-noise ratio in free-form reviews)
2. The role of schema design in shaping review quality
3. Potential for hybrid approaches: free-form prose + structured summary

---

## RQ4: Structural Trade-offs — System Efficiency & Cognitive Cost

### System Efficiency (Scatter Plot)
- **Words:** Free μ = 1,884 vs Struct μ = 793 (↓58%)
- **Latency:** Free μ = 29.1s vs Struct μ = 13.7s (↓53%)
- **Tokens:** Free μ = 2,354 vs Struct μ = 1,113 per review (↓53%)
- **Cost:** $5.65 vs $2.67 per 240-review batch (@$10/1M output tokens)
- Regression lines confirm tight linear coupling between word count and latency

### Cognitive Cost (Violin Plot)
- **Score Variance:** Free σ = 1.27 vs Struct σ = 0.93
- Levene's test: p < 0.001 — variance compression is statistically significant
- Struct's narrower distribution suggests reduced discriminative power (reviews cluster toward the mean)

### The Dual Compression Hypothesis
Structured output imposes a **dual compression**:
1. **Operational compression** (words, tokens, latency, cost) — 53-58% reduction — *beneficial*
2. **Cognitive compression** (score variance) — σ reduced by 27% — *potentially harmful*

### Discussion Points
1. When does variance compression become quality degradation? (Threshold effects)
2. Cost-quality Pareto frontier for LLM review systems
3. Practical recommendation: Use Struct for high-volume triage, Free for final acceptance decisions
4. Schema optimization: Could a richer schema preserve variance while maintaining efficiency?

---

## Cross-Cutting Themes

### The Measurement Paradox
- Free reviews are evaluated by an independent Judge; Struct reviews are self-reported via Pydantic
- This creates an asymmetry: Judge may be harsher/more granular than Pydantic's mechanical `len()`
- Future work: Judge-evaluate both Free AND Struct reviews for true measurement equivalence

### Defense-in-Depth against Injection
- Vision API (PDF→image) eliminates text-layer injection entirely
- Structured output alone is insufficient defense (Firewall Paradox)
- Combined defense: Vision rendering + output validation + anomaly detection on rating distributions

### Generalizability
- N=30 papers from ACL/EMNLP/NeurIPS/ICLR proceedings — representative of top-tier ML venues
- gpt-5.4 as Generator, deepseek-v4-flash as Judge — results may vary with other model pairs
- White-text injection is one of many possible attack vectors (prompt injection in system messages, data poisoning, etc.)

---

## Limitations

1. **Single Generator-Judge pair:** Results bounded to gpt-5.4 + deepseek-v4-flash. Cross-model replication needed.
2. **Binary injection condition:** Only one payload design tested. Gradient of injection strength would reveal dose-response curves.
3. **Within-subjects only:** No between-subjects control group. All papers receive all treatments, which may create carryover effects in LLM state (mitigated by stateless API calls).
4. **Synthetic counterfactuals:** Dycke et al. (2026) perturbations may not capture the full spectrum of real-world logical errors.
5. **English-only:** All papers are English-language ML conference proceedings.
6. **Temperature=0 throughout:** Results represent deterministic worst-case; stochastic decoding (T>0) may show different patterns.

---

## Future Work

1. **Vision API Defense Evaluation:** Quantify injection robustness when PDFs are rendered as images before LLM ingestion
2. **Gradient Injection Strength:** Test payloads of varying explicitness (subtle suggestion → overt command)
3. **Multi-Model Replication:** Extend to Claude, Gemini, Llama families
4. **Schema Optimization:** Design Pydantic schemas that preserve variance while maintaining extraction fidelity
5. **Human Baseline:** Compare LLM reviewer behavior against human reviewers under identical injection conditions
6. **Real-Time Anomaly Detection:** Build monitoring systems that flag suspicious review patterns (sudden MF drops, rating spikes)
