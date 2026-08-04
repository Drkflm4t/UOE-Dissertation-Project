"""Generate blinded validation sample package — v2 with validation_track and simplified schema."""
import pandas as pd
import random
import secrets
import os
from pathlib import Path
import json

# ==============================================================================
# Configuration
# ==============================================================================
PAPER_SEED = secrets.randbits(31)
CONDITION_SEED = secrets.randbits(31)
SHUFFLE_SEED = secrets.randbits(31)
NUM_PAPERS = 5

LOGIC_CONDITIONS = ["blueprint_conclusion", "blueprint_finding", "blueprint_result"]
FORMAT_CONDITIONS = ["active_passive", "british_american", "language_error", "paper_layout"]

OUT_DIR = Path("outputs/manual_validation")
BLIND_DIR = OUT_DIR / "blinded_reviews"
CF_RAW_DIR = Path("outputs/raw_reviews")
INJ_RAW_DIR = Path("outputs/injection_reviews")

manifest_path = OUT_DIR / "manual_validation_sample_manifest.csv"
if manifest_path.exists() and os.environ.get("FORCE_REGENERATE_MANUAL_VALIDATION") != "1":
    raise RuntimeError(
        "Manual-validation sample is locked. Set FORCE_REGENERATE_MANUAL_VALIDATION=1 "
        "only before annotation has started and only when re-blinding is required."
    )

BLIND_DIR.mkdir(parents=True, exist_ok=True)
dev_file = OUT_DIR / "protocol_deviations.md"
if not dev_file.exists():
    dev_file.write_text("# Protocol Deviations\n\n- No deviations recorded yet.\n", encoding="utf-8")

def safe_pid(paper_id):
    return paper_id.replace("%", "_").replace("/", "_").replace("\\", "_")

# ==============================================================================
# Phase 1: Sampling
# ==============================================================================
print("Sampling papers...")
main_df = pd.read_csv("outputs/step3_final_analysis.csv")
all_papers = sorted(main_df['paper_id'].unique())

rng_papers = random.Random(PAPER_SEED)
sampled_papers = rng_papers.sample(all_papers, NUM_PAPERS)

rng_conds = random.Random(CONDITION_SEED)
tasks = []

for paper in sampled_papers:
    chosen_logic = rng_conds.choice(LOGIC_CONDITIONS)
    chosen_format = rng_conds.choice(FORMAT_CONDITIONS)
    # RQ1 Injection Track (4 reviews)
    for cond in ["Original", "Manipulated"]:
        for setup in ["Free", "Struct"]:
            tasks.append({"paper_id": paper, "validation_track": "RQ1",
                          "track": "Injection", "condition": cond + "_PDF", "setup": setup})

    # RQ2 Counterfactual Track (6 reviews)
    for cond in ["Original", chosen_logic, chosen_format]:
        for setup in ["Free", "Struct"]:
            tasks.append({"paper_id": paper, "validation_track": "RQ2",
                          "track": "Counterfactual", "condition": cond, "setup": setup})

print(f"  Total tasks: {len(tasks)}")

# ==============================================================================
# Phase 2: Anonymise & Shuffle
# ==============================================================================
print("Shuffling and anonymising...")
rng_shuffle = random.Random(SHUFFLE_SEED)
rng_shuffle.shuffle(tasks)

manifest_records = []
missing_count = 0

for i, task in enumerate(tasks, start=1):
    review_id = f"MV_{i:03d}"
    pid = task["paper_id"]
    spid = safe_pid(pid)
    cond = task["condition"]
    setup = task["setup"]
    vtrack = task["validation_track"]

    manifest_records.append({
        "review_id": review_id, "validation_track": vtrack,
        "paper_id": pid, "track": task["track"],
        "condition": cond, "setup": setup,
        "coding_order": i, "sample_seed": PAPER_SEED, "condition_seed": CONDITION_SEED,
        "shuffle_seed": SHUFFLE_SEED,
    })

    review_text = f"Review ID: {review_id}\n"
    review_text += f"Validation Track: {vtrack}\n"
    review_text += "=" * 60 + "\n\n"
    found = False
    source_path = None

    if task["track"] == "Counterfactual":
        json_path = CF_RAW_DIR / spid / f"{cond}.json"
        if json_path.exists():
            source_path = json_path
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if setup == "Free":
                content = data.get("prompt_free_text", "")
            else:
                content = json.dumps(data.get("prompt_structured_json", {}), indent=2, ensure_ascii=False)
            review_text += content
            found = True
    else:
        cond_short = cond.replace("_PDF", "")
        suffix = "free" if setup == "Free" else "struct"
        txt_path = INJ_RAW_DIR / spid / f"{cond_short}_{suffix}.txt"
        if txt_path.exists():
            source_path = txt_path
            review_text += txt_path.read_text(encoding="utf-8")
            found = True

    if not found:
        review_text += f"[WARNING: Source file not found for {spid}/{cond} ({setup})]"
        missing_count += 1

    manifest_records[-1]["source_review_path"] = str(source_path) if source_path else ""

    (BLIND_DIR / f"{review_id}.txt").write_text(review_text, encoding="utf-8")

print(f"  {len(tasks) - missing_count}/{len(tasks)} reviews extracted, {missing_count} missing")

# ==============================================================================
# Phase 3: Export CSVs
# ==============================================================================
manifest_df = pd.DataFrame(manifest_records)
manifest_df.to_csv(manifest_path, index=False)
print(f"Manifest locked: {manifest_path}")

annotation_cols = [
    "review_id", "validation_track",
    "human_explicit_rating", "human_inferred_rating_1_10",
    "human_n_strengths", "human_n_weaknesses", "human_n_methodological_flaws",
    "human_injection_compliance_0_10",
]
annotation_df = pd.DataFrame(columns=annotation_cols)
annotation_df["review_id"] = manifest_df["review_id"]
annotation_df["validation_track"] = manifest_df["validation_track"]
annotation_df.to_csv(OUT_DIR / "manual_validation_annotation_sheet.csv", index=False)
print(f"Annotation sheet: {OUT_DIR / 'manual_validation_annotation_sheet.csv'}")

print(f"\nDone! {len(tasks)} blind reviews in {BLIND_DIR}/")
print("DO NOT open manifest.csv until all 50 reviews are coded.")
