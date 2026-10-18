import sys, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

job = json.load(open('job.json'))
prompt = job.get('prompt', '')

model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

messages = [
    {"role": "system", "content": "حلل طلب البرمجة التالي وأجب بصيغة JSON فقط: {\"intent\": \"...\", \"lang\": \"...\", \"summary\": \"...\"}"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

try:
    parsed = json.loads(response[response.find('{'):response.rfind('}')+1])
except:
    parsed = {"intent": "generate", "lang": "unknown", "summary": "تحليل محلي"}

os.makedirs('out', exist_ok=True)
json.dump({"job_id": job['job_id'], **parsed, "model": "PyTorch-Qwen"}, open(f"out/{job['job_id']}.json", "w"), ensure_ascii=False, indent=2)
