import tiktoken
from utils import load_data, url
import torch
from torch.utils.data import Dataset, DataLoader

# prepare input
tokenizer = tiktoken.encoding_for_model('gpt-4')
raw = load_data(url)

# generate tokens
tokens = tokenizer.encode(raw)
print('token size', len(tokens))

# sliding window:context size
context_size = 4
print(f"s1: {tokens[:context_size]}")
print(f"s2:     {tokens[1:context_size+1]}")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []
        # tokenizer the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids) - context_size, stride):
            self.input_ids.append(torch.tensor(token_ids[i: i + context_size]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + context_size + 1]))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, context_size, stride, batch_size, shuffle=False, drop_last=False):
    tokenizer_v1 = tiktoken.encoding_for_model('gpt-4')
    dataset = GPTDatasetV1(txt, tokenizer_v1, context_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last)
    return dataloader

dl = create_dataloader_v1(raw, 4,  4, 5)
print('dataloader size:', len(dl))
data_iter = iter(dl)
print(next(data_iter))
print(next(data_iter))

print('vocab size:', tokenizer.n_vocab)