import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_base_model_and_tokenizer_for_training(model_name: str, device: torch.device):
    """
    Loads a base model with 4-bit quantization and its tokenizer for DPO training.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    print(f'Loading pretrained model on {device} for training...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print('Base model and tokenizer loaded for training.')
    return model, tokenizer

def get_peft_model_for_training(base_model, peft_config: LoraConfig):
    """
    Applies PEFT (LoRA) to the base model for training.
    """
    print('Getting PEFT model for training...')
    model = get_peft_model(base_model, peft_config)
    model.gradient_checkpointing_enable()  # enable checkpointing to reduce memory cost
    model.config.use_cache = False  # disable KV cache to save memory
    print('model.get_input_embeddings().weight.requires_grad: ', model.get_input_embeddings().weight.requires_grad)
    model.print_trainable_parameters()
    return model

def load_dpo_models_and_tokenizer_for_evaluation(base_model_name: str, adapter_ckpt: str, dtype, device_map="auto"):
    """
    Loads the base model (as reference) and the DPO adapter model, along with the tokenizer, for evaluation.
    """
    print("Loading base model for evaluation (reference model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print("Loading DPO adapter model...")
    dpo_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_ckpt,
        torch_dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", max_length=1024)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad_token is set
    print("DPO models and tokenizer loaded for evaluation.")
    return ref_model, dpo_model, tokenizer
