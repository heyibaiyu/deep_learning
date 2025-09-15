from importlib.metadata import version
from ch4_gpt_model_architecture import GPT2Model
from ch4_gpt_model_architecture import generate_text_simple_gpt2, generate_text_simple
import tiktoken
import torch
from torch import nn
from utils import *
from ch2_text_sampling_with_sliding_window import create_dataloader_v1


print('\n----------------- Evaluation generative text model -----------------')

GPT_CONFIG_124M = {
    "vocab_size": 100277,
    "context_length": 256,
    # "context_length": 128,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = get_device_cuda()

torch.manual_seed(123)
model = GPT2Model(GPT_CONFIG_124M)
model.eval()   # disable all the randomness in the model, and disable gradient update

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded


def text_to_token_ids_gpt2(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


start_context = 'Every effort moves you'
tokenizer = tiktoken.encoding_for_model('gpt-4')

def test1():
    token_ids = text_to_token_ids(start_context, tokenizer)
    print('encoded token ids of input', token_ids)

# text = token_ids_to_text(token_ids, tokenizer)
# print('decoded text of input: ', text)


# inference
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"],
)

def test2():
    print(token_ids)
    print(token_ids_to_text(token_ids, tokenizer))

print('\n----------------- Calculate the text generation loss: cross-entropy and perlexity -----------------')

# prepare train/val dataloader
text_data = load_data(url)
token_ids = tokenizer.encode(text_data)
print('token length: ', len(token_ids))

train_ratio = 0.9
split_idx = int(len(text_data) * train_ratio)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
print('train size: ', len(train_data))
print('val size: ', len(val_data))

train_loader = create_dataloader_v1(
    train_data,
    context_size=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    batch_size=2,
    shuffle=True,
    drop_last=True
)

val_loader = create_dataloader_v1(
    val_data,
    context_size=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    batch_size=2,
    shuffle=False,
    drop_last=False
)


for input_ids, target_ids in train_loader:
    print('train_loader shape:', input_ids.shape, target_ids.shape)
    break


# calculate loss of batch
def cal_loss_single_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # print('input_batch shape:', input_batch.shape)
    # print('target_batch shape:', target_batch.shape)

    logits = model(input_batch)
    # print('logits shape:', logits.shape)
    loss = nn.CrossEntropyLoss()(logits.flatten(0, 1), target_batch.flatten())
    return loss


# calculate loss of batch
def cal_loss_single_batch_gpt2(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # print('input_batch shape:', input_batch.shape)
    # print('target_batch shape:', target_batch.shape)

    logits = model(input_batch)
    # print('logits shape:', logits.shape)
    print('target_batch', target_batch)
    print('logits', logits)
    target_batch = target_batch.long()
    loss = nn.CrossEntropyLoss()(logits.logits, target_batch)
    return loss


def test_loss_fun1():
    input_batch = torch.randint(0, 1000, (2, 256), dtype=torch.long)
    target_batch = torch.randint(0, 1000, (2, 256), dtype=torch.long)

    loss = cal_loss_single_batch(input_batch, target_batch, model, "cpu")
    print(loss)

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('inf')
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i , (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_single_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break

    return total_loss / num_batches

def test_loss_fun2():
    torch.manual_seed(123)
    device = get_device_cuda()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print('train_loss: ', train_loss)
    print('val_loss: ', val_loss)
    print('train_perplexity: ', torch.exp(train_loss).item())

print('\n----------------- Training an LLM -----------------')

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # initialize list ot track losses and tokens seen
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # main train loop
    for epoch in range(num_epochs):
        model.train()     # set model to training mode

        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            loss = cal_loss_single_batch(input_batch, target_batch, model, device)

            print(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1
            tokens_seen += input_batch.numel()

            # optional evaluation step
            model.eval()
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device)
                val_loss = calc_loss_loader(val_loader, model, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_token_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (Step {global_step: 06d}):" f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f})")

        # print a sample text after each epoch for inspection
        gemerate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_token_seen

def gemerate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model,
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("decoded text: \n", decoded_text.replace("\n", " "))
    model.train()


def generate_and_print_sample_official_gpt2(model , tokenizer, device, start_context):
    model.eval()
    context_size = model.transformer.wte.embedding_dim
    # print("context_size: ", context_size)
    encoded = text_to_token_ids_gpt2(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple_gpt2(
            model = model,
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("decoded text: \n", decoded_text.replace("\n", " "))
    model.train()

def train_save_load_model():
    torch.manual_seed(123)
    model = GPT2Model(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 4
    train_losses, val_losses, track_token_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        eval_iter=5,
        eval_freq=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    print("train_losses: ", train_losses)
    print("val_losses: ", val_losses)
    # Epoch 4 (Step  00030):Train loss: 5.8051 | val loss: 6.6837)

    print('\n----------------- Decoding strategies to control randomness (Temperature scaling / top k sampling) -----------------')
    print('\n   -----------------  skipped -----------------')

    print('\n----------------- Loading and saving model weights in PyTorch -----------------')

    # save model
    torch.save(model.state_dict(), "./model_gpt2.pth")

    # load model
    model_loaded = GPT2Model(GPT_CONFIG_124M)

    model_loaded.load_state_dict(torch.load("./model_gpt2.pth", map_location=device))
    print("model_loaded: ", model_loaded)

