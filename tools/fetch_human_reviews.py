"""v5: Extract only official review body (Summary → Code Of Conduct), discard author-rebuttal threads."""
import json, sys, re
from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs/manual_validation/human_reviews")
RAW_DIR = OUT_DIR / "raw"
CLEAN_DIR = OUT_DIR / "cleaned"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

main = pd.read_csv("outputs/step3_final_analysis.csv")
all_papers = sorted(main[main["condition"] == "Original"]["paper_id"].unique())
VENUE_MAP = {"ICLR.cc_2025": True, "NeurIPS.cc_2024": True}
on_or = [p for p in all_papers if p.split("%")[0] in VENUE_MAP]

def strip_webpage_chrome(text):
    """Remove everything before the first reviewer block."""
    m = re.search(r'\n\s*(?:Official\s+)?Review', text)
    if m: text = text[m.start():]
    pats = [
        r'(?im)^\s*OpenReview\s*$', r'(?im)^\s*\.net\s*$',
        r'(?im)^\s*Search\s+articles.*$', r'(?im)^\s*Notifications\d*\s*$',
        r'(?im)^\s*Activity\s*$', r'(?im)^\s*Tasks\s*$',
        r'(?im)^\s*back\s+arrow.*$', r'(?im)^\s*Download\s+PDF\s*$',
        r'(?im)^\s*Published:.*$', r'(?im)^\s*Last\s+Modified:.*$',
        r'(?im)^\s*(?:ICLR|NeurIPS)\s+\d{4}.*(?:Poster|Oral|Spotlight).*$',
        r'(?im)^\s*Everyone\s*$', r'(?im)^\s*Revisions\s*$',
        r'(?im)^\s*BibTeX\s*$', r'(?im)^\s*CC\s+BY.*$',
        r'(?im)^\s*Keywords?:.*$', r'(?im)^\s*TL;DR:.*$',
        r'(?im)^\s*Abstract:.*$', r'(?im)^\s*https?://\S+\.pdf\s*$',
        r'(?im)^\s*https?://openreview\.net\S*\s*$',
        r'(?im)^Go\s+to\s+.*homepage.*$',
    ]
    for p in pats: text = re.sub(p, '', text)
    return re.sub(r'^\s*\n+', '', text).strip()

def extract_official_review(text):
    """Keep only Summary → Code Of Conduct; discard header & author-rebuttal threads."""
    m_start = re.search(r'\nSummary\s*:', text)
    if not m_start:
        return text  # fallback
    m_end = re.search(r'\nCode\s+Of\s+Conduct\s*:.*', text[m_start.start():])
    if m_end:
        # Keep through the end of the Code Of Conduct line
        end_pos = m_start.start() + m_end.end()
        return text[m_start.start():end_pos].strip()
    # No explicit Code Of Conduct – take from Summary to end (or to first "Add:" / "Official Comment")
    body = text[m_start.start():]
    m_alt = re.search(r'\n(?:Add|Official\s+Comment|Rebuttal)\s*[:by]', body)
    if m_alt:
        return body[:m_alt.start()].strip()
    return body.strip()

def split_reviews(text):
    """Split on 'Official Review' markers only — not 'Reviewers ...' discussion threads."""
    secs = re.split(r'\n(?=Official\s+Review(?:by|er|\s+of))', text)
    return [s.strip() for s in secs if len(s.strip()) > 100]

def wc(text):
    """Word count excluding section headers and rating/confidence lines."""
    strip = [
        r'(?im)^\s*(?:Official\s+)?Review(?:er)?\s*(?:[A-Z0-9]{3,}|by\s+.*?)?:?\s*$',
        r'(?im)^\s*(?:Summary|Strengths|Weaknesses|Questions|Limitations|Detailed\s+Comments|Comments)\s*:?\s*$',
        r'(?im)^\s*(?:Rating|Confidence|Soundness|Presentation|Contribution|Correctness|Clarity|Reproducibility|Relation\s+to\s+Prior\s+Work|Additional\s+Comments)\s*:?\s*\d.*$',
        r'(?im)^\s*Submitted\s*:.*$', r'(?im)^\s*\d+\s*:?\s*$', r'(?im)^\s*\d+\s*/\s*\d+\s*$',
        # Additional OpenReview UI labels
        r'(?im)^\s*(?:Flag\s+For\s+Ethics\s+Review|Details\s+Of\s+Ethics\s+Concerns|Code\s+Of\s+Conduct)\s*:.*$',
    ]
    t = text
    for p in strip: t = re.sub(p, '', t)
    return len(re.sub(r'\n{3,}', '\n\n', t).split())
    for p in strip: t = re.sub(p, '', t)
    return len(re.sub(r'\n{3,}', '\n\n', t).split())

if "--process" in sys.argv:
    files = list(RAW_DIR.glob("*.txt"))
    if not files: print(f"No .txt in {RAW_DIR}/"); sys.exit(0)
    results = []
    for fp in sorted(files):
        spid = fp.stem
        pid = next((p for p in on_or if p.replace("%","_").replace("/","_") == spid), None)
        if pid is None: print(f"  {fp.name}: unmatched"); continue
        text = strip_webpage_chrome(fp.read_text(encoding="utf-8"))
        secs = split_reviews(text)
        # v5: extract only Summary → Code Of Conduct per review
        secs = [extract_official_review(s) for s in secs]
        secs = [s for s in secs if len(s) > 100 and 'Summary:' in s and wc(s) >= 50]
        wcs = [wc(s) for s in secs]
        # Remove stale files from previous runs for this paper
        for old in CLEAN_DIR.glob(f"{spid}_*"):
            old.unlink()
        for j, s in enumerate(secs):
            (CLEAN_DIR / f"{spid}_r{j+1:02d}.txt").write_text(s, encoding="utf-8")
        (CLEAN_DIR / f"{spid}_meta.json").write_text(json.dumps({
            "paper_id": pid, "safe_id": spid, "n_reviews": len(secs),
            "total_words": sum(wcs), "mean_words": sum(wcs)/max(len(secs),1),
            "word_counts": wcs}, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"paper_id": pid, "n_reviews": len(secs), "total_words": sum(wcs)})
        print(f"  {spid.split('_')[0]:12s}: {len(secs)} reviews, {sum(wcs)} words")
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "human_review_collection_summary.csv", index=False)
    wc_rows = []
    for mf in sorted(CLEAN_DIR.glob("*_meta.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        for rid, w in zip([f"{m['safe_id']}_r{k+1:02d}" for k in range(m['n_reviews'])], m['word_counts']):
            wc_rows.append({"paper_id": m["paper_id"], "review_id": rid, "word_count": w})
    if wc_rows:
        wc_df = pd.DataFrame(wc_rows)
        wc_df.to_csv(OUT_DIR / "human_review_word_counts.csv", index=False)
        ppm = wc_df.groupby("paper_id")["word_count"].mean()
        print(f"\nWords: review mu={wc_df['word_count'].mean():.0f}, paper mu={ppm.mean():.0f}")
    print(f"Done. {len(list(CLEAN_DIR.glob('*_meta.json')))} papers, {int(df['n_reviews'].sum())} reviews in {CLEAN_DIR}/")
else:
    print(f"OpenReview: {len(on_or)}/{len(all_papers)} papers")
    print(f"\n1. Browser: Ctrl+A, Ctrl+C on forum page")
    print(f"2. Paste into: {RAW_DIR}/{{safe_id}}.txt")
    print("3. Run from the project root: python tools/fetch_human_reviews.py --process")
    print(f"   -> strips webpage chrome + Decision/meta per 4.4.2A")
    print(f"   -> flat output: {CLEAN_DIR}/{{safe_id}}_r01.txt ...")
    for pid in on_or:
        spid = pid.replace("%","_").replace("/","_")
        done = "OK" if (RAW_DIR / f"{spid}.txt").exists() else "  "
        print(f"  [{done}] https://openreview.net/forum?id={spid.split('_')[-1]}")
    print(f"\nExisting: {len(list(RAW_DIR.glob('*.txt')))}/{len(on_or)}")
