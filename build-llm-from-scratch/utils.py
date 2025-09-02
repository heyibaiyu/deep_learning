import urllib.request
import os


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
