import re
import tiktoken
from utils import *



class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.int2str = {i: s for i, s in enumerate(self.vocab)}

    def encode(self, text):
        # preprocess
        text = re.split(r'[.,?:;_!"()\']|--|\s', text.lower())
        tokens = [item.strip() for item in text  if item.strip()]
        return [self.vocab[token] for token in tokens]

    def decode(self, ids):
        text = ' '.join([self.int2str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def create_vocab(raw):
    text = re.split(r'[.,?:;_!"()\']|--|\s', raw.lower())
    token_set = sorted(set([item.strip() for item in text if item.strip()]))
    print('vocab size:', len(token_set))
    vocab = {token: id for id, token in enumerate(token_set)}
    return vocab



raw = load_data(url)


s = 'about abruptly, absorbed is a affect! <|endoftext|>'
vocab = create_vocab(raw)
print(vocab)
# tokenizer = Tokenizer(vocab)
# print('\nraw text: ', s)
# encoded = tokenizer.encode(s)
# print('encoded text: ', encoded)
# decoded = tokenizer.decode(encoded)
# print('decoded text: ', decoded)

tokenizer_v2 = tiktoken.encoding_for_model('gpt-4')
# <|endoftext|> is a special token which need to specify in the encode function to avoid exception
encoded = tokenizer_v2.encode(s, allowed_special={'<|endoftext|>'})
print('encoded text: ', encoded)
decoded = tokenizer_v2.decode(encoded)
print('decoded text: ', decoded)