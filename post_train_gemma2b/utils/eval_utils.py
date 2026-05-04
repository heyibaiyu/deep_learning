import torch

def build_labels_left_padding(input_ids, attention_mask, prompt_lens):
    labels = input_ids.clone()
    bsz, seq_total_len = input_ids.shape

    # 1. mask padding
    labels[attention_mask == 0] = -100

    for i in range(bsz):
        seq_len = attention_mask[i].sum().item()
        pad_len = seq_total_len - seq_len

        prompt_start = pad_len
        prompt_end = pad_len + prompt_lens[i]

        # 2. mask prompt tokens
        labels[i, prompt_start:prompt_end] = -100

    return labels


def seq_logprob(model, input_ids, labels):
    """Length-normalized log-prob over response tokens only."""
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

    # avoid divide-by-zero when an example has no response tokens after truncation
    denom = mask.sum(dim=1).clamp(min=1)
    return token_logp.sum(dim=1) / denom
