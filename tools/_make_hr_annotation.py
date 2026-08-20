"""Generate human review annotation sheet for RQ3 aspect coding."""
import json, re, csv
from pathlib import Path

cleaned = Path("outputs/manual_validation/human_reviews/cleaned")
rows = []

for mf in sorted(cleaned.glob("*_meta.json")):
    meta = json.loads(mf.read_text(encoding="utf-8"))
    for k in range(meta["n_reviews"]):
        rid = f"{meta['safe_id']}_r{k+1:02d}"
        txt = (cleaned / f"{rid}.txt").read_text(encoding="utf-8")
        rm = re.search(r"Rating\s*:\s*(\d+)", txt)
        rating = int(rm.group(1)) if rm else ""
        rows.append({
            "review_id": rid,
            "paper_id": meta["paper_id"],
            "human_rating_from_text": rating,
            "human_n_strengths": "",
            "human_n_weaknesses": "",
            "human_n_methodological_flaws": "",
            "notes": "",
        })

out = Path("outputs/manual_validation/human_reviews/human_review_annotation_sheet.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "review_id", "paper_id", "human_rating_from_text",
        "human_n_strengths", "human_n_weaknesses",
        "human_n_methodological_flaws", "notes",
    ])
    w.writeheader()
    w.writerows(rows)

print(f"Done: {len(rows)} rows -> {out}")
for r in rows[:3]:
    print(r)
