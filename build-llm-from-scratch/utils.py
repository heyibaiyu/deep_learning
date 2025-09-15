import urllib.request
import os
import torch
import zipfile
import os
from pathlib import Path


def load_data(url):
    file_path = "the-verdict.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r") as f:
        content = f.read()
        print('file length:', len(content))
    return content

url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")


def get_device_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_device_mps():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    return device

print('get_device_cuda: ', get_device_cuda())
print('get_device_mps: ', get_device_mps())


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def count_trainable_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)