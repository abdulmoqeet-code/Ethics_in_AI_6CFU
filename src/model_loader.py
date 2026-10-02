import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    AutoTokenizer,
    AutoModelForImageTextToText,
    AutoModelForCausalLM
)

# Prevent GPU memory fragmentation issues during batch inference
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def load_vlm(config):
    """
    Initializes Vision-Language or Text-Only Model and Processor/Tokenizer directly from config.
    Supports:
      - 'qwen25_text' : Pure text Causal LM (e.g., Qwen/Qwen2.5-0.5B-Instruct)
      - 'qwen25vl'    : Qwen 2.5 Vision-Language Model
      - 'phi35v'      : Phi-3.5 Vision Instruct
      - 'qwen35'      : Qwen 3.5 / AutoModel ImageTextToText
    """
    model_id = config.get("model_id")
    model_type = config.get("model_type", "").lower()
    hf_token = config.get("hf_token", None)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if not model_id or not model_type:
        raise ValueError("Both 'model_id' and 'model_type' must be specified in the config dictionary.")

    print(f"--- Loading Model: '{model_id}' [Type: {model_type}] on {device} ---")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # 1. Load Model Architecture based on config model_type
    if model_type in ["qwen25_text", "text", "qwen_text"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token
        )
    elif model_type in ["qwen25vl", "qwen2.5-vl"]:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            token=hf_token
        )
    elif model_type in ["phi35v", "phi35_vision", "phi-3.5-vision"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            
            device_map=device_map,
            trust_remote_code=True,
            _attn_implementation="eager",  # Disables broken KV cache methods in custom code
            token=hf_token,
            torch_dtype=torch.bfloat16,
        )
    elif model_type in ["qwen35", "automodel", "qwen3.5"]:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            "Expected one of: 'qwen25_text', 'qwen25vl', 'qwen35', or 'phi35v'."
        )

    # 2. Load AutoProcessor (for VLMs) or AutoTokenizer (for Text-Only)
    if model_type in ["qwen25_text", "text", "qwen_text"]:
        processor = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=False,
            token=hf_token,
            trust_remote_code=True
        )

    # 3. Configure Padding for Tokenizer / Processor
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Fallback CPU placement if CUDA is not active
    if not torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    print(f"Successfully loaded '{model_id}' into memory.\n")
    return model, processor