"""Re-judge 30 Original Free-text reviews to extract strengths, weaknesses, soundness counts."""
import json, os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(".env"), override=False)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-chat")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("ELM_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
print(f"Judge: {JUDGE_MODEL}")

RAW_DIR = Path("outputs/raw_reviews")
paper_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

JUDGE_PROMPT = (
    "Analyze this free-form peer review and extract three metrics:\n"
    "(1) n_strengths: Count the number of distinct strengths or positive aspects mentioned.\n"
    "(2) n_weaknesses: Count the number of distinct weaknesses or criticisms mentioned.\n"
    "(3) n_soundness_issues: Count the number of distinct criticisms that directly question "
    "scientific logic, methodology soundness, or validity of results. "
    "Do NOT count superficial formatting, grammar, or typo complaints.\n"
    'Return ONLY valid JSON: {"n_strengths": int, "n_weaknesses": int, "n_soundness_issues": int}'
)

results = {}
t0 = time.time()
for paper_dir in paper_dirs:
    orig_path = paper_dir / "Original.json"
    if not orig_path.exists():
        continue
    with orig_path.open("r", encoding="utf-8") as f:
        record = json.load(f)
    free_text = record.get("prompt_free_text", "")
    if not free_text:
        continue
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": free_text[:6000]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        results[record["paper_id"]] = {
            "free_n_strengths": int(parsed.get("n_strengths", -1)),
            "free_n_weaknesses": int(parsed.get("n_weaknesses", -1)),
            "free_n_soundness_issues": int(parsed.get("n_soundness_issues", -1)),
        }
        pid = record["paper_id"]
        r = results[pid]
        print(f"  {pid}: s={r['free_n_strengths']}, w={r['free_n_weaknesses']}, so={r['free_n_soundness_issues']}")
    except Exception as e:
        print(f"  FAIL {record['paper_id']}: {e}")

print(f"\nDone in {time.time()-t0:.0f}s. {len(results)}/30 papers extracted.")

out = Path("outputs/rq3_free_aspects.json")
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved to {out}")
