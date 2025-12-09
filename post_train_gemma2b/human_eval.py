import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM

# --- Configuration ---
# You can tune this. 32 is a good starting point for Gemma 2B,
# but try 64 or 128 if you have enough VRAM.
BATCH_SIZE = 32
# ---------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print('using device:', device)

# --- Model Loading (Same) ---
print('loading model...')
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", dtype=torch.float16
).to(device)

adapter_ckpt = "/home/au/jing/gemma2b/checkpoints/checkpoint-7000"
dpo_model = AutoPeftModelForCausalLM.from_pretrained(adapter_ckpt, dtype=torch.float16, device_map=device)

#### debug purpose
for n, p in base_model.named_parameters():
    if 'lora' in n.lower() or 'adapter' in n.lower():
        print(n, p.shape)
#### debug end

print('loading tokenizer...')
# Set padding token for batch inference, essential for batching
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # Use EOS token as pad token

prompts = json.load(open("eval_prompts.json"))
print(f"loaded {len(prompts)} prompts")

# --- Batch Processing Logic ---
results = []
i = 0

# Split all prompts into batches
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing Batches"):
    batch_prompts = prompts[i:i + BATCH_SIZE]

    if i == 0:
        # Display the first prompt of the first batch
        print('\nprompt example:', batch_prompts[0])
    i += 1

    # 1. TOKENIZE THE ENTIRE BATCH
    # padding=True is crucial: it pads the shorter sequences to the length of the longest
    # sequence in the batch, enabling parallel processing.
    inp = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 2. GENERATE OUTPUTS FOR THE ENTIRE BATCH (Base Model)
    # The `attention_mask` handles the padded tokens so the model ignores them.
    out_base = base_model.generate(**inp, max_new_tokens=512, temperature=0.7)

    # 3. GENERATE OUTPUTS FOR THE ENTIRE BATCH (DPO Model)
    out_dpo = dpo_model.generate(**inp, max_new_tokens=512, temperature=0.7)


    # 4. DECODE AND SAVE RESULTS
    for prompt_idx, original_prompt in enumerate(batch_prompts):
        # Extract the relevant output from the batch results
        out_base_single = out_base[prompt_idx]
        out_dpo_single = out_dpo[prompt_idx]

        # Decode, ensuring to cut off the original prompt text from the beginning
        # We need to skip the input tokens for a clean generated response
        input_len = inp['input_ids'][prompt_idx].shape[0]

        res_base = tokenizer.decode(
            out_base_single[input_len:], # Start decoding *after* the input tokens
            skip_special_tokens=True
        )
        res_dpo = tokenizer.decode(
            out_dpo_single[input_len:],
            skip_special_tokens=True
        )

        # Store the result
        results.append({
            "prompt": original_prompt,
            "base": res_base,
            "dpo": res_dpo
        })


json.dump(results, open("manual_eval_candidates.json", "w"), indent=2)
print("\nSaved to manual_eval_candidates.json")