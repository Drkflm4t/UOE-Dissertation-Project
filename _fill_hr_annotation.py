"""Fill human review annotation from user's coding data."""
import pandas as pd
import io

USER_DATA = """review_id,human_n_strengths,human_n_weaknesses,human_n_methodological_flaws
ICLR.cc_2025_8sfc8MwG5v_r01,3,3,6
ICLR.cc_2025_8sfc8MwG5v_r02,4,3,4
ICLR.cc_2025_8sfc8MwG5v_r03,3,1,5
ICLR.cc_2025_8sfc8MwG5v_r04,2,2,4
ICLR.cc_2025_lIVRgt4nLv_r01,2,0,5
ICLR.cc_2025_lIVRgt4nLv_r02,3,1,4
ICLR.cc_2025_lIVRgt4nLv_r03,4,4,0
ICLR.cc_2025_lIVRgt4nLv_r04,3,0,5
ICLR.cc_2025_lLkgj7FEtZ_r01,1,4,3
ICLR.cc_2025_lLkgj7FEtZ_r02,3,5,9
ICLR.cc_2025_lLkgj7FEtZ_r03,3,2,0
ICLR.cc_2025_lLkgj7FEtZ_r04,2,1,4
ICLR.cc_2025_QCDdI7X3f9_r01,2,2,9
ICLR.cc_2025_QCDdI7X3f9_r02,4,3,2
ICLR.cc_2025_QCDdI7X3f9_r03,3,1,5
ICLR.cc_2025_QCDdI7X3f9_r04,4,9,2
NeurIPS.cc_2024_2cQ3lPhkeO_r01,1,3,1
NeurIPS.cc_2024_2cQ3lPhkeO_r02,3,1,2
NeurIPS.cc_2024_2cQ3lPhkeO_r03,3,3,2
NeurIPS.cc_2024_4mxzxYhMuN_r01,4,6,7
NeurIPS.cc_2024_4mxzxYhMuN_r02,3,2,2
NeurIPS.cc_2024_4mxzxYhMuN_r03,3,3,2
NeurIPS.cc_2024_4mxzxYhMuN_r04,3,2,1
NeurIPS.cc_2024_5Hdg5IK18B_r01,3,0,2
NeurIPS.cc_2024_5Hdg5IK18B_r02,3,3,2
NeurIPS.cc_2024_5Hdg5IK18B_r03,3,1,4
NeurIPS.cc_2024_5Hdg5IK18B_r04,3,1,1
NeurIPS.cc_2024_bPuYxFBHyI_r01,5,3,1
NeurIPS.cc_2024_bPuYxFBHyI_r02,4,1,2
NeurIPS.cc_2024_bPuYxFBHyI_r03,3,0,1
NeurIPS.cc_2024_bPuYxFBHyI_r04,3,0,3
NeurIPS.cc_2024_oPFjhl6DpR_r01,4,1,2
NeurIPS.cc_2024_oPFjhl6DpR_r02,3,0,1
NeurIPS.cc_2024_oPFjhl6DpR_r03,3,0,2
NeurIPS.cc_2024_oPFjhl6DpR_r04,3,2,2
NeurIPS.cc_2024_QtYg4g3Deu_r01,3,1,4
NeurIPS.cc_2024_QtYg4g3Deu_r02,3,1,1
NeurIPS.cc_2024_QtYg4g3Deu_r03,3,0,3
NeurIPS.cc_2024_QtYg4g3Deu_r04,3,0,3
NeurIPS.cc_2024_QtYg4g3Deu_r05,3,2,3
NeurIPS.cc_2024_SAZeQV2PtT_r01,3,1,2
NeurIPS.cc_2024_SAZeQV2PtT_r02,3,0,1
NeurIPS.cc_2024_SAZeQV2PtT_r03,2,3,0
NeurIPS.cc_2024_SAZeQV2PtT_r04,3,0,1
NeurIPS.cc_2024_SAZeQV2PtT_r05,2,0,2
NeurIPS.cc_2024_bPuYxFBHyI_r04,5,3,1
NeurIPS.cc_2024_oPFjhl6DpR_r01,4,1,2
NeurIPS.cc_2024_oPFjhl6DpR_r02,3,0,1
NeurIPS.cc_2024_oPFjhl6DpR_r03,3,0,2
NeurIPS.cc_2024_oPFjhl6DpR_r04,3,2,2
NeurIPS.cc_2024_QtYg4g3Deu_r01,3,1,4
NeurIPS.cc_2024_QtYg4g3Deu_r02,3,1,1
NeurIPS.cc_2024_QtYg4g3Deu_r03,3,0,3
NeurIPS.cc_2024_QtYg4g3Deu_r04,3,0,3
NeurIPS.cc_2024_QtYg4g3Deu_r05,3,2,3"""

# Parse user data, take LAST occurrence for duplicates
user_df = pd.read_csv(io.StringIO(USER_DATA))
user_df = user_df.dropna(subset=["review_id"])
# Keep last occurrence per review_id
user_df = user_df.drop_duplicates(subset="review_id", keep="last")
print(f"Parsed: {len(user_df)} unique rows")

# Load existing sheet
sheet = pd.read_csv("outputs/manual_validation/human_reviews/human_review_annotation_sheet.csv")

# Fill
cols = ["human_n_strengths", "human_n_weaknesses", "human_n_methodological_flaws"]
filled = 0
missing = []
for i, row in sheet.iterrows():
    rid = row["review_id"]
    match = user_df[user_df["review_id"] == rid]
    if len(match) == 1:
        for c in cols:
            val = match.iloc[0][c]
            if pd.notna(val):
                sheet.at[i, c] = int(val)
        filled += 1
    else:
        missing.append(rid)

sheet.to_csv("outputs/manual_validation/human_reviews/human_review_annotation_sheet.csv", index=False)
print(f"Filled: {filled}/{len(sheet)}")
if missing:
    print(f"Still empty: {len(missing)} reviews (oPFjhl6DpR + QtYg4g3Deu)")
print("Done.")
