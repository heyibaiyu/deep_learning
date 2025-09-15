import urllib.request

from pathlib import Path
import pandas as pd

from importlib.metadata import version

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

from utils import download_and_unzip_spam_data, count_trainable_parameters, get_device_cuda

from ch5_load_pretrained_openai_weight import load_pretrained_openai_weights
from ch5_pretraining_on_unlabeled_data import train_model_simple

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt


device = get_device_cuda()

def print_pkg_version():
    pkgs = ["matplotlib",  # Plotting library
            "numpy",       # PyTorch & TensorFlow dependency
            "tiktoken",    # Tokenizer
            "torch",       # Deep learning library
            "tensorflow",  # For OpenAI's pretrained weights
            "pandas"       # Dataset loading
           ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def data_inspection(df):
    print('df shape', df.shape)
    print('\n', df.head(2))

    print('\nraw df label distribution: ', df['label'].value_counts())


# map label : ham -> 0, spam -> 1
map_dict = {'ham': 0, 'spam': 1}


def create_balanced_dataset(df):
    # count the instance of 'spam'
    num_spam = df[df['label'] == 'spam'].shape[0]

    # randomly sample same number of 'ham' instances
    ham_subset = df[df['label'] == 'ham'].sample(num_spam, random_state=123)

    combined_df = pd.concat([ham_subset, df[df['label'] == 'spam']])

    combined_df['label'] = combined_df['label'].map(map_dict)
    return combined_df


def random_split(df, train_frac=0.7, val_frac=0.2):
    # shuffle the entire dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    # split the dataframe
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=None, pad_token_id=50256):
        super().__init__()
        self.data = pd.read_csv(csv_file, sep="\t")

        self.encoded_texts = [
            tokenizer.encode(row) for row in self.data['text']
        ]

        if not max_len:
            self.max_len = self._longest_encoded_length__()
        else:
            self.max_len = max_len
            self.encoded_texts = [encoded_text[: self.max_len] for encoded_text in self.encoded_texts]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_len - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        target = self.data['label'][idx]
        return (torch.tensor(encoded_text, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def _longest_encoded_length__(self):
        return max([len(encoded_text) for encoded_text in self.encoded_texts])

def get_data_loader(dataset, batch_size, num_workers=0, drop_last=False):
    torch.manual_seed(123)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches:
        num_batches = min(len(data_loader), num_batches)
    else:
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        target_batch = target_batch.to(device)
        input_batch = input_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch).logits
        logits = logits[:, -1, :]
        predictions = torch.argmax(logits, dim=-1)

        correct_predictions += (predictions == target_batch).sum().item()
        num_examples += input_batch.size(0)

    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device, num_batches=None):
    model.eval()
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch).logits[: , -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.

    if num_batches:
        num_batches = min(len(data_loader), num_batches)
    else:
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device, num_batches)
        total_loss += loss

    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # initialize list ot track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    example_seen, global_step = 0, -1

    # main train loop
    for epoch in range(num_epochs):
        model.train()     # set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            example_seen += input_batch.shape[0]
            global_step += 1


            # optional evaluation step
            model.eval()
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,  device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Epoch {epoch + 1} (Step {global_step: 06d}):" f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f})")

        # calculate accuracy for each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"train_accuracy: {train_accuracy * 100:.2f}% | val_accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, example_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def plot_values(epoches_seen, example_seen, train_values, val_values, label='loss'):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # plot training and validation loss against epochs
    ax1.plot(epoches_seen, train_values, label=f"Training {label}")
    ax1.plot(epoches_seen, val_values, label=f"Validation {label}", linestyle='-.')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # create a second x-axis for example seen
    ax2 = ax1.twiny()  # share same y-axis
    ax2.plot(example_seen, train_values, alpha=0)
    ax2.set_xlabel("Example seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


if __name__ == "__main__":
    print('\n----------------- Finetuning for classification -----------------')

    # download data if not exists
    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    # load downloaded data from local
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["label", "text"])
    data_inspection(df)

    # create balanced df to make sure pos/neg has same count
    balanced_df = create_balanced_dataset(df)
    print('\nbalanced df label distribution: ', balanced_df['label'].value_counts())

    train_df, val_df, test_df = random_split(balanced_df)
    train_df.to_csv('sms_spam_collection_train_val_test/train.csv', sep="\t", index=False)
    val_df.to_csv('sms_spam_collection_train_val_test/val.csv', sep="\t", index=False)
    test_df.to_csv('sms_spam_collection_train_val_test/test.csv', sep="\t", index=False)


    import tiktoken
    tokenize = tiktoken.encoding_for_model("gpt2")
    # print(tokenize.decode([50256]))

    # create dataset
    train_dataset = SpamDataset('sms_spam_collection_train_val_test/train.csv',
                                max_len=None,
                                tokenizer=tokenize)
    print('\nmax length of train dataset', train_dataset.max_len)
    val_dataset = SpamDataset('sms_spam_collection_train_val_test/val.csv', tokenizer=tokenize)
    test_dataset = SpamDataset('sms_spam_collection_train_val_test/test.csv', tokenizer=tokenize)

    # create dataloader
    train_loader = get_data_loader(train_dataset, batch_size=8, num_workers=0, drop_last=True)
    val_loader = get_data_loader(val_dataset, batch_size=8, num_workers=0, drop_last=False)
    test_loader = get_data_loader(test_dataset, batch_size=8, num_workers=0, drop_last=False)

    for input_batch, target_batch in train_loader:
        print('\ninput_batch.shape, target_batch.shape: ', input_batch.shape, target_batch.shape)
        # output: torch.Size([8, 120]) torch.Size([8])
        break


    # load gpt2 model weight from OpenAI
    model = load_pretrained_openai_weights()

    # add a head
    # print(model)
    print(
        '\n All trainable parameters count (initial model): ', count_trainable_parameters(model)
    )

    # freezing the model to make it untrainable
    for param in model.parameters():
        param.requires_grad = False

    print(
        '\n All trainable parameters count (disabled training): ', count_trainable_parameters(model)
    )

    torch.manual_seed(123)
    # update and unfreeze the last head layer
    model.lm_head = nn.Linear(in_features=768, out_features=2, bias=False)

    print(
        '\n All trainable parameters count (updated head layer): ',
        count_trainable_parameters(model)
    )

    # make LayerNorm and last transformer block trainable
    for param in model.transformer.h[-1].parameters():   # last transformer layer
        param.requires_grad = True

    print(
        '\n All trainable parameters count (enabled last transformer layer training): ', count_trainable_parameters(model)
    )

    for param in model.transformer.ln_f.parameters():
        param.requires_grad = True

    print(
        '\n All trainable parameters count (enabled last NormLayer training): ',
        count_trainable_parameters(model)
    )

    # print('classification model structure: ', model)

    input = tokenize.encode("do you have time")
    input = torch.tensor(input).unsqueeze(0)
    out = model(input).logits
    print(out)
    logits = out[:, -1, :]
    print('last row of the output we are focusing: ', logits)

    label = torch.argmax(logits, dim=-1)
    print('label', label.item())

    # initial model accuracy without finetuning
    accu = calc_accuracy_loader(train_loader, model, device, 10)
    print('accu', accu)

    train_loss = calc_loss_loader(train_loader, model, device, 10)
    print('train_loss', train_loss.item())

    # fine tune the model with supervised data

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, example_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        eval_iter=5,
        eval_freq=50,
        start_context="congratulations on the winning!",
        tokenizer=tokenize
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    example_seen_tensor = torch.linspace(0, example_seen, len(train_losses))
    plot_values(epochs_tensor, example_seen_tensor, train_losses, val_losses)

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print('train_accuracy', train_accuracy)
    print('val_accuracy', val_accuracy)
    print('test_accuracy', test_accuracy)





"""
train_accuracy: 100.00% | val_accuracy: 97.50%
Epoch 3 (Step  00300):Train loss: 0.0926 | Val loss: 0.0565)
Epoch 3 (Step  00350):Train loss: 0.0360 | Val loss: 0.0316)
train_accuracy: 100.00% | val_accuracy: 100.00%
Epoch 4 (Step  00400):Train loss: 0.0225 | Val loss: 0.2064)
Epoch 4 (Step  00450):Train loss: 0.0097 | Val loss: 0.1185)
Epoch 4 (Step  00500):Train loss: 0.0040 | Val loss: 0.1421)
train_accuracy: 100.00% | val_accuracy: 95.00%
Epoch 5 (Step  00550):Train loss: 0.0038 | Val loss: 0.0931)
Epoch 5 (Step  00600):Train loss: 0.0114 | Val loss: 0.1107)
Epoch 5 (Step  00650):Train loss: 0.0028 | Val loss: 0.2395)
train_accuracy: 100.00% | val_accuracy: 97.50%
"""