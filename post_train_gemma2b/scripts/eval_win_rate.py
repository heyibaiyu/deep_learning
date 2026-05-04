# evaluate the DPO model win-rate on test set
# compare the reward score of chosen and rejected response

import argparse
from datetime import datetime
import os # Import os for TOKENIZERS_PARALLELISM
import collections # Import collections

import torch
from transformers import AutoTokenizer # Keep AutoTokenizer for direct use in evaluate_dpo_winrate_batched

from utils.data_utils import load_data_ultra_feedback # Updated import path
from utils.model_utils import load_dpo_models_and_tokenizer_for_evaluation # Updated import path
from utils.eval_utils import build_labels_left_padding, seq_logprob # Updated import path


os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizer parallelism globally for this script


def evaluate_dpo_winrate_batched(
    model,
    ref_model,
    tokenizer,
    dataset,
    batch_size=8,
    beta=0.1,
    device="cuda",
):
    model.eval()
    ref_model.eval()

    all_wins = []
    all_margins = []

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
            truncation=True,
        ).to(device)

        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)

        # --- tokenize prompt + response
        enc_chosen = tokenizer(
            prompts,
            chosen,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        enc_rejected = tokenizer(
            prompts,
            rejected,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # --- build masked labels
        labels_chosen = build_labels_left_padding(
            enc_chosen["input_ids"],
            enc_chosen["attention_mask"],
            prompt_lens,
        ).to(device)
        labels_rejected = build_labels_left_padding(
            enc_rejected["input_ids"],
            enc_rejected["attention_mask"],
            prompt_lens,
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
        # print("index:", start, "margin:", margin) # Comment out for cleaner output

        all_wins.append((margin > 0).float())
        all_margins.append(margin)

    overall_win_rate = torch.cat(all_wins).mean().item()
    overall_avg_margin = torch.cat(all_margins).mean().item()

    return overall_win_rate, overall_avg_margin


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPO win-rate on UltraFeedback test split")
    parser.add_argument("--adapter-path", required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--base-model", default="google/gemma-2b-it")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()

    start = datetime.now()
    print(start)
    print("using device:", args.device)

    data = load_data_ultra_feedback(args.split)
    if args.max_examples is not None:
        data = data.select(range(min(args.max_examples, len(data))))

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    ref_model, dpo_model, tokenizer = load_dpo_models_and_tokenizer_for_evaluation(
        base_model_name=args.base_model,
        adapter_ckpt=args.adapter_path,
        dtype=dtype,
        device_map=args.device_map,
    )

    overall_win_rate, overall_avg_margin = evaluate_dpo_winrate_batched(
        model=dpo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=data,
        batch_size=args.batch_size,
        beta=args.beta,
        device=args.device,
    )

    print("\n--- Overall Evaluation Results ---")
    print("adapter path:", args.adapter_path)
    print("DPO win rate:", overall_win_rate)
    print("Reward margin:", overall_avg_margin)

    end = datetime.now()
    print(end)

    duration = end - start
    print("duration", duration)
    print("total minutes", duration.total_seconds() / 60)
    print("total seconds", duration.total_seconds())


if __name__ == "__main__":
    main()
