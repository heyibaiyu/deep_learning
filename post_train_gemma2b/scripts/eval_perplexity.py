import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import math

# --- Config ---
MODEL_ID = "google/gemma-2b-it"
ADAPTER_PATH = "checkpoints/checkpoint-7000"
MAX_LENGTH = 512
SAMPLE_SIZE = 100
BATCH_SIZE = 4 # Use a small batch size for memory efficiency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# 1. Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Filter and sample the texts
texts = [t for t in dataset["text"] if t.strip() != ""]
texts = texts[SAMPLE_SIZE:SAMPLE_SIZE * 2]
print(f"Loaded {len(texts)} samples. Input example: {texts[0][:100]}...")

# 2. Load Model and Tokenizer
print("Loading Model and Tokenizer...")
# Use bfloat16 for stability if supports
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Essential for batching/padding

# 3. Load PEFT Adapter (DPO Model)
print(f"Loading DPO Adapter from {ADAPTER_PATH}...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).eval()

# --- Initialization for Total Accumulation ---
total_nll = 0.0
total_valid_tokens = 0

# 4. Correct Perplexity Calculation Loop
print("Starting Perplexity Calculation...")
# Iterate over the texts in small batches
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing Batches"):
    batch_texts = texts[i:i + BATCH_SIZE]
    
    # Tokenize the current batch
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    # Labels are the input tokens. We set padding tokens to -100
    labels = inputs.input_ids.clone() 
    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        # Forward pass to get the loss (average NLL over valid tokens in the batch)
        outputs = peft_model(**inputs, labels=labels)
        
    # a. Get the MEAN NLL for the batch
    mean_nll_batch = outputs.loss.item()
    
    # b. Count the number of VALID (non-masked) tokens in the batch
    num_valid_tokens_batch = (labels != -100).sum().item()
    
    # c. Calculate the SUM of NLL for the batch and accumulate
    sum_nll_batch = mean_nll_batch * num_valid_tokens_batch
    
    total_nll += sum_nll_batch
    total_valid_tokens += num_valid_tokens_batch


# 5. Final Calculation
if total_valid_tokens > 0:
    # Calculate the overall MEAN NLL across all 50 documents
    overall_mean_nll = total_nll / total_valid_tokens
    
    # Calculate the final MEAN PERPLEXITY: PP = exp(Mean NLL)
    final_perplexity = math.exp(overall_mean_nll)

    print("\n--- Final Perplexity Results ---")
    print(f"Total Valid Tokens Scored: {total_valid_tokens}")
    print(f"Mean Negative Log-Likelihood (NLL): {overall_mean_nll:.4f}")
    print(f"Mean Perplexity: {final_perplexity:.4f}")
else:
    print("Error: No valid tokens were scored. Check dataset filtering and tokenization.")
