import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from alpaca_eval import evaluate
from tqdm import tqdm
from peft import PeftModel
import os
import alpaca_eval.utils as utils

# ==== 1. Load model ====

MODEL = "google/gemma-2b-it"
CKPT = "/home/au/jing/gemma2b/checkpoints/checkpoint-7000"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"Model is on device: {next(model.parameters()).device}")

# Load PEFT adapter (LoRA/DPO weights)
try:
    model.load_adapter(CKPT)
    print("PEFT adapter loaded.")
except Exception as e:
    print("Warning: could not load adapter:", e)

print('running model.eval()')
model.eval()

# ==== 2. Load alpaca_eval.txt ====

print('loading alpaca_eval.txt')
ALPACA_FILE = "/home/au/jing/gemma2b/alpaca_eval.txt"

with open(ALPACA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert to AlpacaEval “instruction-only” format
eval_data = []
for item in raw_data:
    eval_data.append({
        "instruction": item["instruction"],
        "input": "",
    })
print(f"eval data size: {len(eval_data)}")
print('eval data example', eval_data[0])
# ==== 3. Define model inference ====

def run_model(example):
    prompt = example["instruction"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# ==== 4. Run inference ====

print("running inference...")
def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i: i + n]

BATCH_SIZE = 1
MAX_NEW_TOKENS = 512
preds = []
for batch in tqdm(list(batched(eval_data, BATCH_SIZE)), desc="Inference", ncols=100):
    prompts = [ex['instruction'] for ex in batch]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
    preds.extend([{"instruction": prompts[i], "input": "", "output": tokenizer.decode(outs[i], skip_special_tokens=True)} for i in range(len(outs))])
#    preds.append([tokenizer.decode(o, skip_special_tokens=True) for o in outs])
print(f"preds length: {len(preds)}")
print('example pred', preds[0])
with open('alpaca_eval.json', 'w', encoding="utf-8") as f:
    json.dump(preds, f)

# ==== 5. Run AlpacaEval scoring ====

print('running alpacaEval scoring')
results = evaluate(
    model_outputs=preds,
    output_path="alpaca_eval_output/weighted_alpaca_eval_gpt4_turbo",
#    annotators_config="gemma-2b-it",
    annotators_config="weighted_alpaca_eval_gpt4_turbo",
)

print(results)
