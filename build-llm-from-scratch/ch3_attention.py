import torch
from matplotlib import pyplot as plt
from torch import nn

# ----------------- Section 3.1
# ----------------- A simple self-attention mechanism
# ----------------- Compute attention weight
print('\n----------------- section 1 -----------------')
inputs = torch.tensor([[0.43, 0.15, 0.89],
                      [0.55, 0.87, 0.66],
                      [0.57, 0.85, 0.64],
                      [0.43, 0.15, 0.89],
                      [0.43, 0.15, 0.89],
                      [0.43, 0.15, 0.89]])

# assume query is inputs[1], calculate the attention factor (dot product) between query and all other tokens
input_query = inputs[1]
print('input first query:', input_query)

attn_score = torch.empty(inputs.shape[0])
for idx, value in enumerate(inputs):
    attn_score[idx] = torch.dot(input_query, value)

print('attn_score:', attn_score)

# normalization
attn_score_norm = attn_score / attn_score.sum()
print('attn_score_norm:', attn_score_norm)
print('attn_score_norm sum:', attn_score_norm.sum())

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_score_softmax = softmax(attn_score)
print('attn_score_softmax:', attn_score_softmax)

attn_score_softmax_default = torch.softmax(attn_score, dim=0)
print('attn_score_softmax_default:', attn_score_softmax_default)

# generate context vector
def generate_context_vector(queries, atte_weight):
    context_vector = torch.empty(input_query.shape[0])
    for idx_tmp, value_tmp in enumerate(queries):
        context_vector += value_tmp * atte_weight[idx_tmp]
    return context_vector

context_vector = generate_context_vector(inputs, attn_score_softmax)
print('context_vector:', context_vector)


# ----------------- Section 3.2
# ----------------- A simple self-attention mechanism
# ----------------- Generalize attention weight calculation

print('\n----------------- section 2 -----------------')
attn_score_matrix = torch.empty(inputs.shape[0], inputs.shape[0])
# a naive implementation
for i, value_i in enumerate(inputs):
    for j, value_j in enumerate(inputs):
        attn_score_matrix[i, j] = torch.dot(value_i, value_j)
# print('attn_score_matrix:', attn_score_matrix)

# instead, we use matrix multiplication
attn_score_matrix = inputs @ inputs.T
att_weights =  torch.softmax(attn_score_matrix, dim=1)
# print('att_weights:', att_weights)

all_context_vector = att_weights @ inputs
print('all_context_vector:', all_context_vector)

# ----------------- Section 3.3
# ----------------- implementing self attention with trainable weights
print('\n----------------- section 3 -----------------')
print('inputs shape:', inputs.shape)
x_2 = inputs[1]
print('x_2 shape:', x_2.shape)
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
print('W_query shape', W_query.shape)

query_2 = x_2 @ W_query
print('query_2 shape:', query_2.shape)

keys = inputs @ W_key
print('keys shape:', keys.shape)
values = inputs @ W_value

keys_2 = keys[1]
attn_score_22 = torch.dot(query_2, keys_2)
print('attn_score_22:', attn_score_22)

attn_score_2 = query_2 @ keys.T
print('attn_score_2 shape:', attn_score_2.shape)
print('attn_score_2:', attn_score_2)

d_k = keys.shape[1]
attn_weight_2 = torch.softmax(attn_score_2 / d_k ** 0.5, dim=-1)
print('attn_weight_2 shape:', attn_weight_2.shape)

# compute context vector
context_vector_2 = attn_weight_2 @ values
print('context_vector_2 shape:', context_vector_2.shape)
print('context_vector_2:', context_vector_2)


# self attention compact class v1
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # compute q, k, v -> attn_score -> attn_weight -> context vector
        queries = x @ self.W_query  # assume input dim: 6 x 2, d_in: 3, d_out: 2, then queries dim: 6 x 2
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_score = queries @ keys.T    # dim: 6 x 6
        d_k = keys.shape[1]              # dim: 2
        attn_weight = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        context_vector = attn_weight @ values
        return context_vector

torch.manual_seed(123)
self_attention_v1 = SelfAttentionV1(d_in, d_out)
out = self_attention_v1(inputs)
print('self_attention_v1 out shape:', out.shape)
print('self_attention_v1 out:', out)


# self attention compact class v2
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)

    def forward(self, x):
        # compute q, k, v -> attn_score -> attn_weight -> context vector
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_score = queries @ keys.T    # dim: 6 x 6
        d_k = keys.shape[1]              # dim: 2
        attn_weight = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        context_vector = attn_weight @ values
        return context_vector

torch.manual_seed(123)
self_attention_v2 = SelfAttentionV2(d_in, d_out)
out = self_attention_v2(inputs)
# note: v2 result is different from v1. The reason is randomization difference within nn.Linear
print('self_attention_v2 out shape:', out.shape)
print('self_attention_v2 out:', out)


# ----------------- Section 3.4
# ----------------- hidden future words with causal attention
print('\n----------------- section 4 -----------------')
print('    ------------- hidden future words with causal attention')

# create diagonal masked attn_weight so that tokens afterward are unseen
    # approach 1: attn_score * torch.tril(1s)
t = torch.ones(6, 6)
t = torch.tril(torch.ones(6, 6))
print('tril ones', t)

torch.manual_seed(123)
t1 = torch.rand(6, 6)
print('t1 random', t1)
t2 = t * t1
print('t2 tril random', t2)

    # approach 2: mask_fill attn_score with diagonal tril with -inf, then apply softmax. it becomes zero
t1=t1.masked_fill(torch.triu(torch.ones(6, 6).bool(), diagonal=1), -torch.inf)
print('t1 mask tril random', t1)
# print(torch.triu(torch.ones(6, 6).bool()))
t1_softmax = torch.softmax(t1, dim=-1)
print('t1_softmax :', t1_softmax)


# ----------------- Section 3.5
# -----------------  Dropout mask
print('\n----------------- section 5 -----------------')
print('    ------------- drop out mask ')

torch.manual_seed(123)
layer_dropout = torch.nn.Dropout(p=0.2)
example = torch.ones(6, 6)
out = layer_dropout(example)
print('drop out mask out', out)

# -----------------  compact into attention class
print('\n------------- compact into causal attention class')

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout=0, context_length=0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # compute q, k, v -> attn_score -> attn_weight -> context vector
        batch_size, token_len, input_dim = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_score = queries @ keys.transpose(1, 2)    # dim: 6 x 6
        d_k = keys.shape[1]              # dim: 2
        attn_score.masked_fill_(self.mask.bool()[:token_len, :token_len], -torch.inf)
        attn_weight = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        self.dropout(attn_weight)
        context_vector = attn_weight @ values
        return context_vector

batch = torch.stack((inputs, inputs))
print('input batch data shape:', batch.shape)
context_length = batch.shape[1]
causal_attention = CausalAttention(d_in, d_out, dropout=0, context_length=context_length)
out = causal_attention(batch)
print('causal attention out shape:', out.shape)


# ----------------- Section 3.6
# -----------------  multihead attention
print('\n----------------- section 6: multi-head attention -----------------')

class MultiHeadAttentionV1(nn.Module):
    def __init__(self, d_in, d_out, heads_num, dropout=0, context_length=0):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, dropout=dropout, context_length=context_length) for _ in range(heads_num)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
multi_head_attention = MultiHeadAttentionV1(d_in, d_out, heads_num=5, context_length=context_length)
out = multi_head_attention(batch)
print('multi_head_attention out shape:', out.shape)
# print('multi_head_attention out:', out)


# ----------------- Section 3.7
# -----------------  more efficient multihead attention implementation with weight splits
print('\n----------------- section 6: multi-head attention -----------------')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec