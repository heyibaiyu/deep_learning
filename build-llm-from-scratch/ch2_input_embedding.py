import torch
from ch2_text_sampling_with_sliding_window import *

# convert token ids to token embedding

# set hyperparameter

# input = torch.tensor([1, 3])
vocab_size = 100277    # tokenizer.n_vocab
context_length = 4
embedding_dim = 256

torch.manual_seed(42)

dataloader = create_dataloader_v1(raw, context_length,  context_length, 8)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print('input shape', inputs.shape)

# 1. create token embedding layer
token_emb = torch.nn.Embedding(vocab_size, embedding_dim)
print('token emb weight shape:', token_emb.weight.shape)

token_emb_out = token_emb(inputs)
print('token emb out shape:', token_emb_out.shape)

# 2. create pos embedding layer
# same weight for each sample: one dim less than input
pos_emb = torch.nn.Embedding(context_length, embedding_dim)
pos_emb_out = pos_emb(torch.arange(context_length))
print('pos embedding shape:', pos_emb_out.shape)

# 3. add token embedding with pos embedding as final embedding
input_embedding_out = token_emb_out + pos_emb_out
print('input embedding shape:', input_embedding_out.shape)

