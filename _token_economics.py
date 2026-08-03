"""Token Economics: count output tokens from raw reviews using tiktoken."""
import json, tiktoken
from pathlib import Path

encoder = tiktoken.encoding_for_model("gpt-4o")
raw_dir = Path("outputs/raw_reviews")

total_free = 0
total_struct = 0
file_count = 0

for jf in raw_dir.rglob("*.json"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    free_text = data.get("prompt_free_text", "")
    struct_data = data.get("prompt_structured_json", {})
    struct_text = json.dumps(struct_data, ensure_ascii=False) if struct_data else ""
    total_free += len(encoder.encode(free_text))
    total_struct += len(encoder.encode(struct_text))
    file_count += 1

reduction = (total_free - total_struct) / total_free * 100
price_per_1m = 10.0  # USD per 1M output tokens
cost_free = total_free / 1_000_000 * price_per_1m
cost_struct = total_struct / 1_000_000 * price_per_1m

print(f"Token Economics (N={file_count} Counterfactual reviews)")
print(f"{'='*55}")
print(f"  Free:     {total_free:>8,} tokens  (~${cost_free:.2f} @ $10/1M)")
print(f"  Struct:   {total_struct:>8,} tokens  (~${cost_struct:.2f} @ $10/1M)")
print(f"  Reduction: {reduction:.1f}%  (saved ~${cost_free - cost_struct:.2f})")
print(f"  Avg Free:   {total_free / file_count:,.0f} tokens/review")
print(f"  Avg Struct: {total_struct / file_count:,.0f} tokens/review")
