"""Quick test: can Chat Completions accept PDF files?"""
import os, base64, json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env')
client = OpenAI(api_key=os.getenv('ELM_API_KEY'), base_url=os.getenv('ELM_BASE_URL','https://api.openai.com/v1'))
model = os.getenv('ELM_MODEL','gpt-5.4')

pdf_path = sorted(Path('outputs/manipulated_pdfs').glob('*/original.pdf'))[0]
print(f'Model: {model}  |  PDF: {pdf_path.name}  |  Size: {pdf_path.stat().st_size} bytes')

with open(pdf_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

# Test 1: data URL
try:
    resp = client.chat.completions.create(
        model=model, max_tokens=50,
        messages=[{'role':'user','content':[
            {'type':'text','text':'What is the title of this paper? Reply with just the title.'},
            {'type':'file','file_data': f'data:application/pdf;base64,{b64}'}
        ]}]
    )
    print(f'[1] file + data URL => OK: {resp.choices[0].message.content[:200]}')
except Exception as e:
    err = str(e)
    short = err[:200] if len(err) > 200 else err
    print(f'[1] file + data URL => FAIL: {short}')

# Test 2: image_url with data URL (vision path)
try:
    resp = client.chat.completions.create(
        model=model, max_tokens=50,
        messages=[{'role':'user','content':[
            {'type':'text','text':'What is the title? Reply with just the title.'},
            {'type':'image_url','image_url':{'url': f'data:application/pdf;base64,{b64}','detail':'low'}}
        ]}]
    )
    print(f'[2] image_url + data URL => OK: {resp.choices[0].message.content[:200]}')
except Exception as e:
    err = str(e)
    short = err[:200] if len(err) > 200 else err
    print(f'[2] image_url + data URL => FAIL: {short}')

# Test 3: upload file, then use file_id in content
try:
    with open(pdf_path, 'rb') as f:
        file = client.files.create(file=f, purpose='user_data')
    resp = client.chat.completions.create(
        model=model, max_tokens=50,
        messages=[{'role':'user','content':[
            {'type':'text','text':'What is the title? Reply with just the title.'},
            {'type':'file','file_id': file.id}
        ]}]
    )
    print(f'[3] file_id content => OK: {resp.choices[0].message.content[:200]}')
    client.files.delete(file.id)
except Exception as e:
    err = str(e)
    short = err[:200] if len(err) > 200 else err
    print(f'[3] file_id content => FAIL: {short}')

# Test 4: plain text extraction (baseline: does this model even work?)
try:
    resp = client.chat.completions.create(
        model=model, max_tokens=50,
        messages=[{'role':'user','content':'Say "hello from gpt-5.4"'}]
    )
    print(f'[0] basic call => OK: {resp.choices[0].message.content.strip()}')
except Exception as e:
    print(f'[0] basic call => FAIL: {str(e)[:150]}')

print('\nDone.')
