import torch
from nltk.draw import cfg
from torch import nn
import tiktoken
from matplotlib import pyplot as plt
import math
from ch3_attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 100277,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        # use a placeholder for transformer block
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        # use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        token_emb = self.token_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



t1 = 'this is a cute'
t2 = 'hello file del ok'

tokenizer = tiktoken.encoding_for_model('gpt-4')
token_ids_1 = torch.tensor(tokenizer.encode(t1))
token_ids_2 = torch.tensor(tokenizer.encode(t2))
batch_data = torch.stack((token_ids_1, token_ids_2))
print('input data shape', batch_data.shape)
print('vocab size', tokenizer.n_vocab)
dummy_model = DummyGPTModel(cfg=GPT_CONFIG_124M)
out = dummy_model(batch_data)
print('dummy model output shape', out.shape)


print('\n----------------- layer normalization -----------------')
torch.manual_seed(123)
small_example = torch.randn(2, 5)
print('small example:', small_example)

layer = nn.Sequential(nn.Linear(small_example.shape[1], 6), nn.ReLU())
out = layer(small_example)
print('layer output:', out)
print('mean:', out.mean(dim=-1, keepdim=True), 'var:', out.var())
out1 = (out - out.mean(dim=-1, keepdim=True)) / torch.sqrt(out.var(dim=-1, keepdim=True))
print('normalized mean:', out1.mean(dim=-1), 'var:', out1.var(dim=-1))


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.offset = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return x * self.scale + self.offset

layer_norm = LayerNorm(6)
out = layer_norm(torch.randn(2, 6))
print('layer norm output', out)
print('normalized mean:', out.mean(dim=-1, keepdim=True), 'var:', out.var(dim=-1))

print('\n----------------- a feed forward network with GELU activations ------------------')

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        a2 = self.sqrt_2_over_pi * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1 + torch.tanh(a2))

# # visualize the GELU function
# gelu = nn.GELU()
# inputs = torch.arange(-2, 2, 0.1, dtype=torch.float32)
# out = gelu(inputs)
# out2 = torch.nn.functional.relu(inputs)
# plt.plot(inputs, out, label='GELU')
# plt.plot(inputs, out2, label='ReLU')
# plt.legend()
# plt.grid(True)
# plt.show()

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        return self.layer(x)

inputs = torch.randn(2, 3, 768)   # 2: batch size, 3: number of tokens, 768: token dimension
ffs = FeedForward(cfg=GPT_CONFIG_124M)
out = ffs(inputs)
print('feed forward output shape', out.shape)
print('feed forward output', out)

print('\n----------------- adding shortcut connections/residual connections -----------------')

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        i = 0
        for layer in self.layers:
            layer_out = layer(x)
            print(f"Layer {i} output: {layer_out}")
            i += 1
            if self.use_shortcut and x.shape == layer_out.shape:
                x = layer_out + x
            else:
                x = layer_out

        return x

def print_gradients(model, x):
    # forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # calculate loss
    loss = nn.MSELoss()
    loss = loss(output, target)

    # backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        # print(name, param)
        if 'weight' in name:
            # print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

def test1():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    print('input shape:', sample_input.shape)
    torch.manual_seed(123)
    model1 = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model1, sample_input)


print('\n----------------- connection attention and linear layers in a transformer block -----------------')

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg['emb_dim'],
                                      d_out=cfg['emb_dim'],
                                      context_length=cfg['context_length'],
                                      num_heads=cfg['n_heads'],
                                      dropout=cfg['drop_rate'])
        self.ff =  FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        # Dropout is stochastic during training: each call randomly zeroes some elements.
        # so it's good to use same instance in different places. It doesn't mean they share weight/param
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

def test2():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    model = TransformerBlock(cfg=GPT_CONFIG_124M)
    out = model(x)
    print(f"Input shape: {x.shape}, output shape: {out.shape}")

print('\n----------------- Code GPT model architecture -----------------')


class GPT2Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        token_emb = self.token_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def test3():
    torch.manual_seed(123)
    gpt_model = GPT2Model(cfg=GPT_CONFIG_124M)

    # prepare batch input with real text
    # t1 = 'this is a cute'
    # t2 = 'hello file del ok'
    # tokenizer = tiktoken.encoding_for_model('gpt-4')
    # token_ids_1 = torch.tensor(tokenizer.encode(t1))
    # token_ids_2 = torch.tensor(tokenizer.encode(t2))
    # batch_data = torch.stack((token_ids_1, token_ids_2))
    print('input data shape', batch_data.shape)
    print('vocab size', tokenizer.n_vocab)

    out = gpt_model(batch_data)
    print('gpt model output shape', out.shape)

    total_params = sum(p.numel() for p in gpt_model.parameters())
    print(f'total params: {total_params:,}')    # 239,840,256, note there is no weight sharing here

    model_token_emb_weight = gpt_model.token_emb.weight
    print(model_token_emb_weight.shape)
    weight2 = gpt_model.out_head.weight
    print(weight2.shape)

    print('model size after sharing weight: ', total_params - weight2.shape[0] * weight2.shape[1])


print('\n----------------- Generate text -----------------')

# inference
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size :]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_simple_gpt2(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size :]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits.logits
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def run_inference(model, start_context):
    encoded = tokenizer.encode(start_context)
    print('encoded', encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print('encoded_tensor shape', encoded_tensor.shape)

    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 6,
        context_size = GPT_CONFIG_124M['context_length']
    )

    print('out', out)

    l = (out.squeeze(0)).tolist()
    decoded = tokenizer.decode(l)
    print('decoded', decoded)

if __name__ == '__main__':
    start_context = 'Hello, I am'
    # run_inference(gpt_model, batch_data)