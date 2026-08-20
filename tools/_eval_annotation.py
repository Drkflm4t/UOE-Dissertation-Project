"""Evaluate human annotation vs Free Judge automated metrics."""
import pandas as pd, numpy as np, scipy.stats as stats

ann = pd.read_csv("outputs/manual_validation/manual_validation_annotation_sheet.csv")
man = pd.read_csv("outputs/manual_validation/manual_validation_sample_manifest.csv")

merged = ann.merge(man[["review_id","paper_id","track","condition","setup"]], on="review_id")

cf = pd.read_csv("outputs/step3_final_analysis.csv")
free_inj = pd.read_csv("outputs/step2_pdf_track_free_rated.csv")

free_merged = merged[merged["setup"] == "Free"].copy()

rows = []
for _, r in free_merged.iterrows():
    pid, cond, track = r["paper_id"], r["condition"], r["track"]
    auto_rating = auto_ns = auto_nw = auto_nmf = auto_comp = None
    if track == "Counterfactual":
        match = cf[(cf["paper_id"] == pid) & (cf["condition"] == cond)]
        if len(match):
            auto_rating = match.iloc[0]["free_extracted_rating"]
            auto_ns = match.iloc[0]["free_n_strengths"]
            auto_nw = match.iloc[0]["free_n_weaknesses"]
            auto_nmf = match.iloc[0]["free_n_methodological_flaws"]
    else:
        match = free_inj[(free_inj["paper_id"] == pid) & (free_inj["condition"] == cond)]
        if len(match):
            auto_rating = match.iloc[0]["extracted_rating"]
            auto_ns = match.iloc[0]["extracted_n_strengths"]
            auto_nw = match.iloc[0]["extracted_n_weaknesses"]
            auto_nmf = match.iloc[0]["extracted_n_methodological_flaws"]
            auto_comp = match.iloc[0]["injection_compliance_score"]
    rows.append({
        "review_id": r["review_id"], "track": track,
        "human_rating": r["human_inferred_rating_1_10"],
        "auto_rating": auto_rating,
        "human_ns": r["human_n_strengths"], "auto_ns": auto_ns,
        "human_nw": r["human_n_weaknesses"], "auto_nw": auto_nw,
        "human_nmf": r["human_n_methodological_flaws"], "auto_nmf": auto_nmf,
        "human_comp": r["human_injection_compliance_0_10"], "auto_comp": auto_comp,
    })

cmp = pd.DataFrame(rows)

def fmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

valid = cmp.dropna(subset=["human_rating", "auto_rating"])
print(f"Free reviews with both human & auto: {len(valid)} / {len(cmp)}")
print()

print(f"{'Metric':12s}  {'r':>6s}  {'MAE':>6s}  {'Human mu':>9s}  {'Auto mu':>9s}  {'Bias':>7s}  {'p(bias=0)':>10s}")
print("-" * 75)

for label, hcol, acol in [
    ("Rating", "human_rating", "auto_rating"),
    ("Strengths", "human_ns", "auto_ns"),
    ("Weaknesses", "human_nw", "auto_nw"),
    ("MFs", "human_nmf", "auto_nmf"),
]:
    sub = cmp[[hcol, acol]].dropna()
    r, p = stats.pearsonr(sub[hcol], sub[acol])
    mae = np.mean(np.abs(sub[hcol] - sub[acol]))
    diff = sub[hcol] - sub[acol]
    _, pt = stats.ttest_1samp(diff, 0)
    print(f"{label:12s}  {r:+6.3f}  {mae:6.2f}  {sub[hcol].mean():9.2f}  {sub[acol].mean():9.2f}  {np.mean(diff):+7.2f}  {fmt(pt):>10s}")

print()
sub_c = cmp[cmp["track"] == "Injection"][["human_comp", "auto_comp"]].dropna()
if len(sub_c) > 1:
    r, p = stats.pearsonr(sub_c["human_comp"], sub_c["auto_comp"])
    mae = np.mean(np.abs(sub_c["human_comp"] - sub_c["auto_comp"]))
    print(f"Compliance : r={r:+.3f} ({fmt(p)}), MAE={mae:.2f}, human={sub_c['human_comp'].mean():.2f}, auto={sub_c['auto_comp'].mean():.2f}")

# Also compare Structured
print("\n=== Structured: Human vs Pydantic Self-Report ===")
struct_merged = merged[merged["setup"] == "Struct"].copy()
sr_rows = []
for _, r in struct_merged.iterrows():
    pid, cond, track = r["paper_id"], r["condition"], r["track"]
    auto_rating = auto_ns = auto_nw = auto_nmf = None
    if track == "Counterfactual":
        match = cf[(cf["paper_id"] == pid) & (cf["condition"] == cond)]
        if len(match):
            auto_rating = match.iloc[0]["rating_1_10"]
            auto_ns = match.iloc[0]["n_strengths"]
            auto_nw = match.iloc[0]["n_weaknesses"]
            auto_nmf = match.iloc[0]["n_methodological_flaws"]
    else:
        ps = pd.read_csv("outputs/step2_pdf_track_structured_results.csv")
        match = ps[(ps["paper_id"] == pid) & (ps["condition"] == cond)]
        if len(match):
            auto_rating = match.iloc[0]["rating_1_10"]
            auto_ns = match.iloc[0]["n_strengths"]
            auto_nw = match.iloc[0]["n_weaknesses"]
            auto_nmf = match.iloc[0]["n_methodological_flaws"]
    sr_rows.append({
        "human_rating": r["human_inferred_rating_1_10"],
        "auto_rating": auto_rating,
        "human_ns": r["human_n_strengths"], "auto_ns": auto_ns,
        "human_nw": r["human_n_weaknesses"], "auto_nw": auto_nw,
        "human_nmf": r["human_n_methodological_flaws"], "auto_nmf": auto_nmf,
    })

sr = pd.DataFrame(sr_rows)
for label, hcol, acol in [
    ("Rating", "human_rating", "auto_rating"),
    ("Strengths", "human_ns", "auto_ns"),
    ("Weaknesses", "human_nw", "auto_nw"),
    ("MFs", "human_nmf", "auto_nmf"),
]:
    sub = sr[[hcol, acol]].dropna()
    if len(sub) < 2: continue
    r, p = stats.pearsonr(sub[hcol], sub[acol])
    mae = np.mean(np.abs(sub[hcol] - sub[acol]))
    diff = sub[hcol] - sub[acol]
    _, pt = stats.ttest_1samp(diff, 0)
    print(f"{label:12s}  r={r:+.3f} ({fmt(p)}), MAE={mae:.2f}, human={sub[hcol].mean():.2f}, auto={sub[acol].mean():.2f}, bias={np.mean(diff):+.2f} ({fmt(pt)})")
