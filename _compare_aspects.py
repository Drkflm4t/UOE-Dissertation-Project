"""Compare human vs LLM aspect counts."""
import pandas as pd, numpy as np

ha = pd.read_csv("outputs/manual_validation/human_reviews/human_review_annotation_sheet.csv")
ha["total"] = ha["human_n_strengths"] + ha["human_n_weaknesses"] + ha["human_n_methodological_flaws"]

s3 = pd.read_csv("outputs/step3_final_analysis.csv")
orig = s3[s3["condition"] == "Original"]

# Free = judge-extracted; Structured = Pydantic self-report
fs, fw, fm = orig["free_n_strengths"], orig["free_n_weaknesses"], orig["free_n_methodological_flaws"]
ft = fs + fw + fm

ss, sw, sm = orig["n_strengths"], orig["n_weaknesses"], orig["n_methodological_flaws"]
st = ss + sw + sm

hs, hw, hm = ha["human_n_strengths"], ha["human_n_weaknesses"], ha["human_n_methodological_flaws"]
ht = ha["total"]

print(f"{'':25s} {'HUMAN':>10s} {'FREE':>10s} {'STRUCT':>10s}")
print(f"{'':25s} {'(N=45)':>10s} {f'(N={len(orig)})':>10s} {f'(N={len(orig)})':>10s}")
print("=" * 60)
for label, h, f, s in [
    ("Strengths", hs, fs, ss),
    ("Weaknesses", hw, fw, sw),
    ("Methodological Flaws", hm, fm, sm),
    ("TOTAL", ht, ft, st),
]:
    print(f"{label:25s} {h.mean():>8.1f}  {f.mean():>8.1f}  {s.mean():>8.1f}")

print("-" * 60)
hmf = hm.sum() / ht.sum() * 100
fmf = fm.sum() / ft.sum() * 100
smf = sm.sum() / st.sum() * 100
print(f"{'MF / Total':25s} {hmf:>7.0f}%  {fmf:>7.0f}%  {smf:>7.0f}%")

hs_ratio = hs.sum() / ht.sum() * 100
fs_ratio = fs.sum() / ft.sum() * 100
ss_ratio = ss.sum() / st.sum() * 100
print(f"{'Strength / Total':25s} {hs_ratio:>7.0f}%  {fs_ratio:>7.0f}%  {ss_ratio:>7.0f}%")

# Word count comparison
hwc = pd.read_csv("outputs/manual_validation/human_reviews/human_review_word_counts.csv")
print(f"\n{'Word count':25s} {hwc.word_count.mean():>8.0f}  {orig.free_words.mean():>8.0f}  {orig.structured_words.mean():>8.0f}")
print(f"{'Aspects / 100 words':25s} {ht.sum()/hwc.word_count.sum()*100:>7.1f}  {ft.sum()/orig.free_words.sum()*100:>7.1f}  {st.sum()/orig.structured_words.sum()*100:>7.1f}")

# Correlation: rating vs total aspects
print(f"\n=== Rating vs Aspects (Human) ===")
ha["rating"] = ha["human_rating_from_text"]
for cat in ["human_n_strengths", "human_n_weaknesses", "human_n_methodological_flaws", "total"]:
    r = ha["rating"].corr(ha[cat])
    print(f"  rating ~ {cat:30s}: r={r:+.2f}")

# Distribution of MF  
print(f"\n=== MF distribution ===")
print(f"Human MF=0: {(hm==0).sum()/45*100:.0f}%")
print(f"Free  MF=0: {(fm==0).sum()/len(orig)*100:.0f}%")
print(f"Struct MF=0: {(sm==0).sum()/len(orig)*100:.0f}%")
