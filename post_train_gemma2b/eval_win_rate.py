# evaluate the DPO model win-rate on test set
# compare the reward score of chosen and rejected response

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import log_softmax
from data_util import load_data_ultra_feedback
from datetime import datetime

start = datetime.now()
print(start)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('using device:', device)


def build_labels_left_padding(input_ids, attention_mask, prompt_lens):
    labels = input_ids.clone()
    B, T = input_ids.shape

    # 1. mask padding
    labels[attention_mask == 0] = -100

    for i in range(B):
        seq_len = attention_mask[i].sum().item()
        pad_len = T - seq_len

        prompt_start = pad_len
        prompt_end = pad_len + prompt_lens[i]

        # 2. mask prompt tokens
        labels[i, prompt_start:prompt_end] = -100

    return labels


def seq_logprob(model, input_ids, labels):
    """
    Length-normalized log-prob over response tokens only.
    Safe for CUDA.
    """
    with torch.no_grad():
        logits = model(input_ids).logits

    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    labels = labels[:, 1:]

    # mask prompt tokens
    mask = labels != -100

    # replace -100 with a safe index (e.g., 0)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    token_logp = logp.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_logp = token_logp * mask

    return token_logp.sum(dim=1) / mask.sum(dim=1)


def evaluate_dpo_winrate_batched(
    model,
    ref_model,
    tokenizer,
    dataset,
    batch_size=8,
    beta=0.1,
    device="cuda"
):
    model.eval()
    ref_model.eval()

    wins = []
    margins = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]

        prompts = list(batch["prompt"])
        chosen = list(batch["chosen"])
        rejected = list(batch["rejected"])

        # --- tokenize prompt alone (for prompt length)
        enc_prompt = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)

        # --- tokenize prompt + response
        enc_chosen = tokenizer(
            prompts,
            chosen,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        enc_rejected = tokenizer(
            prompts,
            rejected,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # --- build masked labels
        labels_chosen = build_labels_left_padding(
            enc_chosen["input_ids"],
            enc_chosen["attention_mask"],
            prompt_lens
        ).to(device)
        labels_rejected = build_labels_left_padding(
            enc_rejected["input_ids"],
            enc_rejected["attention_mask"],
            prompt_lens
        ).to(device)

        # --- policy model log-probs
        logp_c = seq_logprob(model, enc_chosen["input_ids"], labels_chosen)
        logp_r = seq_logprob(model, enc_rejected["input_ids"], labels_rejected)

        # --- reference model log-probs
        ref_logp_c = seq_logprob(ref_model, enc_chosen["input_ids"], labels_chosen)
        ref_logp_r = seq_logprob(ref_model, enc_rejected["input_ids"], labels_rejected)

        # --- DPO implicit rewards
        reward_c = beta * (logp_c - ref_logp_c)
        reward_r = beta * (logp_r - ref_logp_r)

        margin = reward_c - reward_r
        print('index: ', start, 'margin:', margin)

        wins.append((margin > 0).float())
        margins.append(margin)

    win_rate = torch.cat(wins).mean().item()
    avg_margin = torch.cat(margins).mean().item()

    return win_rate, avg_margin


def load_model(dpo_model_root, dpo_model_name):
    print('loading model...')
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it", dtype=torch.float16, device_map="auto")
    adapter_ckpt = dpo_model_root + dpo_model_name
    dpo_model = AutoPeftModelForCausalLM.from_pretrained(adapter_ckpt, dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", padding_side="left", max_length=1024)
    print('model loaded')
    return base_model, dpo_model, tokenizer


def load_data():
    return load_data_ultra_feedback('test')


data = load_data()
# dpo_model_root = "/home/au/jing/gemma2b/checkpoints/"
# dpo_model_name = 'checkpoint-7000'
dpo_model_root = "/home/au/jing/gemma2b/gemma2b_1/dpo-gemma2b_4/"
dpo_model_name = 'checkpoint-4=6000'

ref_model, dpo_model, tokenizer = load_model(dpo_model_root, dpo_model_name)

win_rate, margin = evaluate_dpo_winrate_batched(
    model=dpo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=data,
    batch_size=4,   # tune for GPU memory
    beta=0.1,
    device=device
)

print('model name: ', dpo_model_name)
print("DPO win rate:", win_rate)
print("Reward margin:", margin)

end = datetime.now()
print(end)

duration = end - start
print('duration', duration)
print('total minutes', duration.total_seconds() / 60)
print('total seconds', duration.total_seconds())
