# !pip install bitsandbytes
# !pip install trl
# !pip install -U bitsandbytes

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
import sys # Keep sys for stdout/stderr redirection
from utils.data_utils import load_data_ultra_feedback # Updated import path

from utils.model_utils import load_base_model_and_tokenizer_for_training, get_peft_model_for_training # Updated import path
from huggingface_hub import login
login(new_session=False)


logfile = open("logs/train.log", "w") # Updated log file path
sys.stdout = logfile # Keep stdout redirection
sys.stderr = logfile # Keep stderr redirection

"""# Get base model and prepare for training"""

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(device)

model_name = 'google/gemma-2b-it'

# Load base model and tokenizer
base_model, tokenizer = load_base_model_and_tokenizer_for_training(model_name, device)

# Setup LoRA config
peft_config = LoraConfig(
    r=16, # lower r to reduce memory
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bias='none',
    task_type='CAUSAL_LM')

# Apply PEFT to the base model
model = get_peft_model_for_training(base_model, peft_config)

"""# Prepare training data"""

dataset = load_data_ultra_feedback()

"""# Train model"""

training_args = DPOConfig(
    output_dir="checkpoints/dpo-gemma2b_4",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    max_steps=7000,
    beta=0.1,       # Important: strength of preference alignment
    max_length=512,
    max_prompt_length=256
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
logfile.close()
