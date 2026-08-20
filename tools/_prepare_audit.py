"""Extract RQ2 counterfactual edits and RQ3 best pair data."""
import json
from pathlib import Path
import pandas as pd

cf_root = Path("data/cf_datasets")
maps = {
    "blueprint_result": "blueprint_result_picf",
    "blueprint_conclusion": "blueprint_conclusion_picf",
}

out = Path("outputs/manual_validation/rq2_rq3_audit_data.txt")
lines = []
lines.append("=" * 70)
lines.append("RQ2: COUNTERFACTUAL EDIT TEXTS (5 Logic cases)")
lines.append("=" * 70)

cases = [
    ("2024.acl%2024.acl-long.375", "blueprint_result", "MV_001(Struct) vs MV_044(Free)"),
    ("2024.emnlp%2024.emnlp-main.758", "blueprint_conclusion", "MV_007(Struct) vs MV_046(Free)"),
    ("2024.emnlp%2024.emnlp-main.1123", "blueprint_result", "MV_013(Struct) vs MV_039(Free)"),
    ("NeurIPS.cc_2024%2cQ3lPhkeO", "blueprint_conclusion", "MV_033(Struct) vs MV_018(Free)"),
    ("2024.acl%2024.acl-long.741", "blueprint_result", "MV_038(Struct) vs MV_040(Free)"),
]

for i, (pid, cond, mvs) in enumerate(cases, 1):
    folder = maps[cond]
    jp = cf_root / folder / f"{pid}.json"
    if not jp.exists():
        jp = cf_root / folder / f"{pid.replace('%', '_')}.json"
    if not jp.exists():
        lines.append(f"\n### Case {i}: {pid} ({cond}) - FILE NOT FOUND")
        continue

    d = json.loads(jp.read_text(encoding="utf-8"))
    changes = d.get("changes", {})
    lines.append(f"\n{'─' * 60}")
    lines.append(f"### Case {i}: {pid.split('%')[-1][:30]}")
    lines.append(f"Condition: {cond}")
    lines.append(f"Reviews: {mvs}")
    lines.append(f"OPERATION: {changes.get('operation', 'N/A')}")
    bt = changes.get("break_target", {})
    if bt:
        # blueprint_result structure
        rs = bt.get("result_summary", "")
        if rs:
            lines.append(f"ORIGINAL RESULT: {rs[:250]}")
        # blueprint_conclusion structure
        cs = bt.get("conclusion_summary", "")
        if cs:
            lines.append(f"ORIGINAL CONCLUSION: {cs[:250]}")
        br = changes.get("break_result", {})
        if br:
            nkf = br.get("negated_key_fact", "")
            if nkf:
                lines.append(f"NEGATED FACT: {nkf[:250]}")
            er = br.get("edited_result", "") or br.get("conclusion_summary", "")
            if er:
                lines.append(f"EDITED: {er[:250]}")
            dc = br.get("detailed_changes", "")
            if dc:
                lines.append(f"DETAILS: {dc[:300]}")

# RQ3
lines.append("")
lines.append("=" * 70)
lines.append("RQ3: BEST ORIGINAL PAIR — 2024.emnlp-main.758 (gap +9)")
lines.append("=" * 70)

pid = "2024.emnlp%2024.emnlp-main.758"
cf_df = pd.read_csv("outputs/step3_final_analysis.csv")
row = cf_df[(cf_df["paper_id"] == pid) & (cf_df["condition"] == "Original")]
if len(row):
    r = row.iloc[0]
    lines.append(f"Auto: Free {r['free_n_weaknesses']}w + {r['free_n_methodological_flaws']}mf = {int(r['free_n_weaknesses'])+int(r['free_n_methodological_flaws'])} total")
    lines.append(f"Auto: Struct {r['n_weaknesses']}w + {r['n_methodological_flaws']}mf = {int(r['n_weaknesses'])+int(r['n_methodological_flaws'])} total")
    lines.append("")
    man = pd.read_csv("outputs/manual_validation/manual_validation_annotation_sheet.csv")
    man_m = pd.read_csv("outputs/manual_validation/manual_validation_sample_manifest.csv")
    m = man.merge(man_m[["review_id","paper_id","condition","setup"]], on="review_id")
    for _, rm in m[(m["paper_id"]==pid) & (m["condition"].isin(["Original"]))].iterrows():
        lines.append(f"Human {rm['setup']:6s} ({rm['review_id']}): {int(rm['human_n_weaknesses'])}w + {int(rm['human_n_methodological_flaws'])}mf = {int(rm['human_n_weaknesses'])+int(rm['human_n_methodological_flaws'])} total")

out.write_text("\n".join(lines), encoding="utf-8")
print(f"Written: {out}")
print("\n".join(lines))
