import json, os
from llama_cpp import Llama

job = json.load(open('job.json'))
prompt = job.get('prompt', '')

llm = Llama(model_path="models/qwen.gguf", n_ctx=2048, n_threads=4, verbose=False)

sys_prompt = "أنت محلل برمجي. أجب بصيغة JSON فقط: {\"intent\": \"...\", \"lang\": \"...\", \"summary\": \"...\"}"
res = llm.create_chat_completion(
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
    max_tokens=150, temperature=0.1
)

response = res['choices'][0]['message']['content']
try:
    parsed = json.loads(response[response.find('{'):response.rfind('}')+1])
except:
    parsed = {"intent": "generate", "lang": "unknown", "summary": "تحليل محلي"}

os.makedirs('out', exist_ok=True)
json.dump({"job_id": job['job_id'], **parsed, "model": "Qwen-1.5B-GGUF"}, open(f"out/{job['job_id']}.json", "w"), ensure_ascii=False, indent=2)
