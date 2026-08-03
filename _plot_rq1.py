"""RQ1: Injection Track ATE — Free vs Struct (all int)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Data ──
pf = pd.read_csv("outputs/step2_pdf_track_free_rated.csv")
pf_ok = pf[pf["ok"] == True]
fo = pf_ok[pf_ok["condition"] == "Original_PDF"].set_index("paper_id")["extracted_rating"].dropna()
fm = pf_ok[pf_ok["condition"] == "Manipulated_PDF"].set_index("paper_id")["extracted_rating"].dropna()
f_ix = fo.index.intersection(fm.index)

ps = pd.read_csv("outputs/step2_pdf_track_structured_rated.csv")
ps_ok = ps[ps["ok"] == True]

# ── Metric helper ──
def compute_ate(df, col):
    so = df[df["condition"] == "Original_PDF"].set_index("paper_id")[col].dropna()
    sm = df[df["condition"] == "Manipulated_PDF"].set_index("paper_id")[col].dropna()
    ix = so.index.intersection(sm.index)
    d = sm.loc[ix].values - so.loc[ix].values
    return float(np.mean(d)), float(stats.sem(d)), stats.ttest_rel(sm.loc[ix], so.loc[ix])[1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel A: Free vs Struct Rating ATE ──
ax = axes[0]
ate_free, se_free, p_free = compute_ate(pf_ok, "extracted_rating")
ate_struct, se_struct, p_struct = compute_ate(ps_ok, "rating_1_10")

bars = ax.bar(["Free (Judge)", "Struct (Self)"], [ate_free, ate_struct],
              yerr=[se_free, se_struct], capsize=8, width=0.45,
              color=["#ED7D31", "#4472C4"], edgecolor="white", linewidth=1.2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Δ Rating (Manipulated − Original)")
ax.set_title("RQ1: Injection Effect on Rating (int-aligned)")

for bar, val, p in zip(bars, [ate_free, ate_struct], [p_free, p_struct]):
    va = "bottom" if val >= 0 else "top"
    offset = 0.08 if val >= 0 else -0.08
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{val:+.2f}\np={p:.1e}", ha="center", va=va, fontsize=11, fontweight="bold")

# Gap annotation
gap = ate_free - ate_struct
mid_y = (ate_free + ate_struct) / 2
ax.annotate("", xy=(0.8, ate_struct + 0.05), xytext=(0.8, ate_free - 0.05),
            arrowprops=dict(arrowstyle="<->", color="#D62728", lw=2))
ax.text(0.95, mid_y, f"Gap = {gap:.2f}", ha="left", va="center",
        fontsize=12, fontweight="bold", color="#D62728")

# ── Panel B: Struct Self — All Metrics ATE ──
ax = axes[1]
metrics = [
    ("Rating",     "rating_1_10"),
    ("Strengths",  "n_strengths"),
    ("Weaknesses", "n_weaknesses"),
    ("M-Flaws",    "n_methodological_flaws"),
]
labels, deltas, ses, ps_vals = [], [], [], []
for label, col in metrics:
    d, se, p = compute_ate(ps_ok, col)
    labels.append(label)
    deltas.append(d)
    ses.append(se)
    ps_vals.append(p)

colors = ["#4472C4" if d >= 0 else "#D62728" for d in deltas]
bars = ax.bar(labels, deltas, yerr=ses, capsize=8, width=0.5,
              color=colors, edgecolor="white", linewidth=1.2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Δ (Manipulated − Original)")
ax.set_title("Struct Self: All Metrics ATE")

for bar, val, p in zip(bars, deltas, ps_vals):
    va = "bottom" if val >= 0 else "top"
    offset = 0.10 if val >= 0 else -0.10
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{val:+.1f}", ha="center", va=va, fontsize=11, fontweight="bold",
            color=bar.get_facecolor())

plt.tight_layout()
fig.savefig("figures/rq1_ate_int_aligned.png", dpi=300, bbox_inches="tight")
print("Saved: figures/rq1_ate_int_aligned.png")
print(f"\nFree  ATE = {ate_free:+.2f} (p={p_free:.2e})")
print(f"Struct ATE = {ate_struct:+.2f} (p={p_struct:.2e})")
print(f"Gap        = {gap:.2f}")
