"""Gather all stats needed for Chapter 4 P0/P1 — v2."""
import pandas as pd, numpy as np, json
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "outputs"

# ── RQ1: Injection data ──
fr = pd.read_csv(BASE / "step2_pdf_track_free_rated.csv")
sr = pd.read_csv(BASE / "step2_pdf_track_structured_rated.csv")

fo = fr[fr["condition"] == "Original_PDF"].set_index("paper_id")
fm = fr[fr["condition"] == "Manipulated_PDF"].set_index("paper_id")
so = sr[sr["condition"] == "Original_PDF"].set_index("paper_id")
sm = sr[sr["condition"] == "Manipulated_PDF"].set_index("paper_id")

common_pids = fo.index.intersection(fm.index).intersection(so.index).intersection(sm.index)
print(f"N injection pairs: {len(common_pids)}")

print("\n" + "=" * 70)
print("TABLE 4.1 — RQ1 Attack Effects")
print("=" * 70)

for label, fc, sc in [
    ("Rating", "extracted_rating", "rating_1_10"),
    ("MF", "extracted_n_methodological_flaws", "n_methodological_flaws"),
    ("Strengths", "extracted_n_strengths", "n_strengths"),
    ("Weaknesses", "extracted_n_weaknesses", "n_weaknesses"),
    ("Compliance", "injection_compliance_score", "injection_compliance_score"),
]:
    for tlabel, df_o, df_m, col in [
        ("Free", fo, fm, fc),
        ("Struct", so, sm, sc),
    ]:
        delta = df_m[col] - df_o[col]
        valid = delta.dropna()
        if len(valid) == 0:
            print(f"  {label:12s} {tlabel:7s} NO DATA")
            continue
        ci = stats.t.interval(0.95, len(valid)-1, loc=valid.mean(), scale=stats.sem(valid))
        print(f"  {label:12s} {tlabel:7s} orig={df_o[col].mean():.2f} inj={df_m[col].mean():.2f} D={valid.mean():+.2f} CI=[{ci[0]:+.2f},{ci[1]:+.2f}]")

print("\n--- Attenuation Contrasts (FreeD - StructD) ---")
for label, fc, sc in [
    ("Rating", "extracted_rating", "rating_1_10"),
    ("MF", "extracted_n_methodological_flaws", "n_methodological_flaws"),
    ("Strengths", "extracted_n_strengths", "n_strengths"),
    ("Weaknesses", "extracted_n_weaknesses", "n_weaknesses"),
    ("Compliance", "injection_compliance_score", "injection_compliance_score"),
]:
    fdelta = fm[fc] - fo[fc]
    sdelta = sm[sc] - so[sc]
    contrast = fdelta - sdelta
    valid = contrast.dropna()
    if len(valid) == 0:
        print(f"  {label:12s} NO DATA")
        continue
    t, p = stats.ttest_rel(fdelta.dropna(), sdelta.dropna())
    ci = stats.t.interval(0.95, len(valid)-1, loc=valid.mean(), scale=stats.sem(valid))
    print(f"  {label:12s} Contrast={valid.mean():+.2f} CI=[{ci[0]:+.2f},{ci[1]:+.2f}] t={t:.2f} p={p:.4f}")

# ── RQ2: Counterfactual data ──
print("\n" + "=" * 70)
print("TABLE 4.2 — RQ2 Discriminability")
print("=" * 70)

s3 = pd.read_csv(BASE / "step3_final_analysis.csv")
cf = s3[s3["condition"] != "Original"]
# Aggregate by paper_id within each group (papers appear in multiple counterfactual types)
logic = cf[cf["group"] == "Logic-Perturbed"].groupby("paper_id").mean(numeric_only=True)
surface = cf[cf["group"] == "Format-Perturbed"].groupby("paper_id").mean(numeric_only=True)

common = logic.index.intersection(surface.index)
logic = logic.loc[common]
surface = surface.loc[common]
print(f"N RQ2 pairs: {len(common)}")

for metric, fc, sc in [
    ("Rating", "free_extracted_rating", "rating_1_10"),
    ("MF", "free_n_methodological_flaws", "n_methodological_flaws"),
]:
    for tlabel, col in [("Free", fc), ("Struct", sc)]:
        delta = logic[col] - surface[col]
        valid = delta.dropna()
        t, p = stats.ttest_rel(logic[col].dropna(), surface[col].dropna())
        ci = stats.t.interval(0.95, len(valid)-1, loc=valid.mean(), scale=stats.sem(valid))
        print(f"  {metric:8s} {tlabel:7s} Logic={logic[col].mean():.2f} Surface={surface[col].mean():.2f} D={valid.mean():+.2f} CI=[{ci[0]:+.2f},{ci[1]:+.2f}] t={t:.2f} p={p:.4f}")

# Holm correction across 4 tests
pvals = []
test_info = []
for metric, fc, sc in [
    ("Rating", "free_extracted_rating", "rating_1_10"),
    ("MF", "free_n_methodological_flaws", "n_methodological_flaws"),
]:
    for tlabel, col in [("Free", fc), ("Struct", sc)]:
        delta = logic[col] - surface[col]
        valid = delta.dropna()
        t, p = stats.ttest_rel(logic[col].dropna(), surface[col].dropna())
        pvals.append(p)
        test_info.append((metric, tlabel))

n = len(pvals)
ranks = np.argsort(pvals)
holm_reject = np.zeros(n, dtype=bool)
for i, r in enumerate(ranks):
    holm_reject[r] = pvals[r] < 0.05 / (n - i)

print("\nHolm-adjusted:")
for i, (metric, tlabel) in enumerate(test_info):
    print(f"  {metric:8s} {tlabel:7s} p={pvals[i]:.4f} Holm={'SIG' if holm_reject[i] else 'ns'}")

# ── RQ3 stats ──
print("\n" + "=" * 70)
print("TABLE 4.3 — RQ3 Evidence Triangulation")
print("=" * 70)

orig_s3 = s3[s3["condition"] == "Original"]

print("\nPanel A: N=30 Operational Profile (Original only)")
for label, s, w, m, wc_col in [
    ("Free", "free_n_strengths", "free_n_weaknesses", "free_n_methodological_flaws", "free_words"),
    ("Struct", "n_strengths", "n_weaknesses", "n_methodological_flaws", "structured_words"),
]:
    sv = orig_s3[s]; wv = orig_s3[w]; mv = orig_s3[m]
    tot = sv + wv + mv
    print(f"  {label:7s} S={sv.mean():.1f} W={wv.mean():.1f} MF={mv.mean():.1f} Total={tot.mean():.1f} Words={orig_s3[wc_col].mean():.0f}")
    print(f"          S%={sv.sum()/tot.sum()*100:.0f}% W%={wv.sum()/tot.sum()*100:.0f}% MF%={mv.sum()/tot.sum()*100:.0f}%")

print("\nPanel B: N=5 Common Coding (Auxiliary subset)")
mv_path = BASE / "manual_validation" / "manual_validation_annotation_sheet.csv"
mv = pd.read_csv(str(mv_path))
for track in ["RQ1", "RQ2"]:
    sub = mv[mv["validation_track"] == track]
    if len(sub):
        stot = sub["human_n_strengths"] + sub["human_n_weaknesses"] + sub["human_n_methodological_flaws"]
        print(f"  {track}: N={len(sub)} S={sub.human_n_strengths.mean():.1f} W={sub.human_n_weaknesses.mean():.1f} MF={sub.human_n_methodological_flaws.mean():.1f} Total={stot.mean():.1f}")

print("\nPanel C: N=11 Human External Benchmark (11 papers, 45 reviews)")
ha = pd.read_csv(str(BASE / "manual_validation" / "human_reviews" / "human_review_annotation_sheet.csv"))
hwc = pd.read_csv(str(BASE / "manual_validation" / "human_reviews" / "human_review_word_counts.csv"))
ht = ha["human_n_strengths"] + ha["human_n_weaknesses"] + ha["human_n_methodological_flaws"]
print(f"  Reviews={len(ha)}, Papers={ha.paper_id.nunique()}")
print(f"  S={ha.human_n_strengths.mean():.1f} W={ha.human_n_weaknesses.mean():.1f} MF={ha.human_n_methodological_flaws.mean():.1f} Total={ht.mean():.1f}")
print(f"  S%={ha.human_n_strengths.sum()/ht.sum()*100:.0f}% W%={ha.human_n_weaknesses.sum()/ht.sum()*100:.0f}% MF%={ha.human_n_methodological_flaws.sum()/ht.sum()*100:.0f}%")
print(f"  Words: mu={hwc.word_count.mean():.0f} med={hwc.word_count.median():.0f} SD={hwc.word_count.std():.0f} range=[{hwc.word_count.min()},{hwc.word_count.max()}]")

# ── RQ4 stats ──
print("\n" + "=" * 70)
print("TABLE 4.4 — RQ4 Trade-offs")
print("=" * 70)

print(f"\nPanel A: N=240 Full (30 papers x 8 conditions)")
print(f"  Free words:   mu={s3.free_words.mean():.0f} med={s3.free_words.median():.0f} SD={s3.free_words.std():.0f}")
print(f"  Struct words: mu={s3.structured_words.mean():.0f} med={s3.structured_words.median():.0f} SD={s3.structured_words.std():.0f}")
print(f"  Ratio (Struct/Free): {s3.structured_words.mean()/s3.free_words.mean()*100:.0f}%")
print(f"  Free latency:   mu={s3.free_latency_sec.mean():.1f}s med={s3.free_latency_sec.median():.1f}s")
print(f"  Struct latency: mu={s3.structured_latency_sec.mean():.1f}s med={s3.structured_latency_sec.median():.1f}s")
print(f"  Latency ratio:  {s3.structured_latency_sec.mean()/s3.free_latency_sec.mean()*100:.0f}%")

fd = s3.groupby("paper_id")["free_extracted_rating"].std()
sd = s3.groupby("paper_id")["rating_1_10"].std()
from scipy.stats import levene
lstat, lp = levene(fd.dropna(), sd.dropna())
print(f"  Free rating sigma:   mu={fd.mean():.2f} med={fd.median():.2f}")
print(f"  Struct rating sigma: mu={sd.mean():.2f} med={sd.median():.2f}")
print(f"  Levene: stat={lstat:.2f} p={lp:.4f}")

print(f"\nPanel B: 11-paper Human Length Benchmark")
print(f"  Reviews: {len(hwc)}, Papers: {ha.paper_id.nunique()}")
print(f"  Words: mu={hwc.word_count.mean():.0f} med={hwc.word_count.median():.0f} SD={hwc.word_count.std():.0f}")
print(f"  Range: [{hwc.word_count.min()}, {hwc.word_count.max()}]")
print(f"  Human/Free ratio:   {hwc.word_count.mean()/s3.free_words.mean()*100:.0f}%")
print(f"  Human/Struct ratio: {hwc.word_count.mean()/s3.structured_words.mean()*100:.0f}%")

print("\n=== ALL STATS GATHERED ===")
