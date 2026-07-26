import os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv('.env')
client = OpenAI(api_key=os.getenv('ELM_API_KEY'), base_url=os.getenv('ELM_BASE_URL'))
model = os.getenv('ELM_MODEL')

md_path = sorted(Path('data/papers').rglob('paper.md'))[0]
md = md_path.read_text(encoding='utf-8')[:50000]
print(f'Model: {model} | Paper: {md_path.parent.name} | {len(md)} chars')

pf = 'You are an expert academic reviewer. Please write a full peer review for the paper below.\n\nPaper Content:\n' + md
t0 = time.time()
r = client.chat.completions.create(model=model, temperature=0, messages=[{'role':'user','content':pf}])
t1 = time.time()
print(f'Free: {t1-t0:.1f}s | {len(r.choices[0].message.content)} chars')

ps = 'You are an expert academic reviewer. Evaluate the paper along 6 dimensions: summary, strengths, weaknesses, soundness, rating (1-10), confidence (1-5). Return strictly as JSON.\n\nPaper Content:\n' + md
t0 = time.time()
r2 = client.beta.chat.completions.parse(model=model, temperature=0, messages=[{'role':'user','content':ps}], response_format={'type':'json_object'})
t2 = time.time()
import json
parsed = json.loads(r2.choices[0].message.content)
print(f'Struct: {t2-t0:.1f}s | rating={parsed.get("rating_1_10", "?")}')

est = (t1-t0 + t2-t0) * 240
print(f'\nMain Track: 240 rows x {(t1-t0+t2-t0):.0f}s/row = {est:.0f}s = {est/60:.1f} min')
print(f'PDF Track: 120 calls x 4.5s = {120*4.5:.0f}s = {120*4.5/60:.1f} min')
print(f'TOTAL: {(est+120*4.5)/60:.1f} min')
