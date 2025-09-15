
import tiktoken

from ch5_pretraining_on_unlabeled_data import generate_and_print_sample_official_gpt2
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_device_cuda

device = get_device_cuda()

print('\n----------------- Load OpenAI pretrained weights and evaluation -----------------')
def load_pretrained_openai_weights():
    print("Loading GPT2LMHeadModel...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("Model loaded successfully!")
    model.to(device)
    model.eval()
    return model

def generate_text(model, start_context):
    # do not use gpt-4 tokenizer, otherwise the output will be arbitrary
    # tokenizer = tiktoken.encoding_for_model('gpt-4')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    generate_and_print_sample_official_gpt2(model, tokenizer, device, start_context)
    # output: Every effort moves you forward.
    # The first step is to understand the importance of your work.
    # The second step is to understand the importance of your work.
    # The third step is to understand the importance of your work.
    # The fourth step is

if __name__ == "__main__":
    model = load_pretrained_openai_weights()
    start_contexts = ['Every effort moves you',
                      '1 + 1 =',
                      '1+1=2, 2+3=5, 5+2=7, 9+0=9, 10+20=30, 5+4=9, 9+1=']
    """ OUTPUT: 
    decoded text: 
     Every effort moves you forward.  The first step is to understand the importance of your work.  The second step is to understand the importance of your work.  The third step is to understand the importance of your work.  The fourth step is
    decoded text: 
     1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 + 1 = 1 +
    decoded text: 
     1+1=2, 2+3=5, 5+2=7, 9+0=9, 10+20=30, 5+4=9, 9+1=10, 10+2=11, 10+3=12, 10+4=13, 10+5=14, 10+6=15, 10+7=16, 10+8=17, 10+9=18,

    """
    for start_context in start_contexts:
        generate_text(model, start_context)

