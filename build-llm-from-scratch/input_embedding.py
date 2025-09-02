import torch

# convert token ids to token embedding

input = torch.tensor([1, 3])
print('input shape', input.shape)

# 1. create token embedding layer
# set hyperparameter
context_length = 4
embedding_dim = 256

torch.manual_seed(42)
token_emb = torch.nn.Embedding(context_length, embedding_dim)
print('initial emb weight:', token_emb.weight)
print('emb weight shape:', token_emb.weight.shape)

token_emb_out = token_emb(input)
print('token emb shape:', token_emb_out.shape)

# 2. create pos embedding layer
pos_emb = torch.nn.Embedding(context_length, embedding_dim)
pos_emb_out = pos_emb(input)
print('pos embedding shape:', pos_emb_out.shape)

# 3. add token embedding with pos embedding as final embedding
input_embedding_out = token_emb_out + pos_emb_out
print('input embedding shape:', input_embedding_out.shape)

