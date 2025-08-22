# To test different models
llama = 0  # 0 - Qwen2.5-7B-Instruct; 1 - Llama-3-8B-Instruct

if llama == 1:
    MODEL = "Llama-3-8B-Instruct"
    HF_BASE_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_ADAPTER_PATH = "Pk3112/medmcqa-lora-llama3-8b-instruct"  # Hub repo id
else:
    MODEL = "Qwen2.5-7B-Instruct"
    HF_BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
    HF_ADAPTER_PATH = "Pk3112/medmcqa-lora-qwen2.5-7b-instruct"  # Hub repo id

# App settings
MAX_TURNS = 20
BUDGET_CHARS = 12000
PAGE_TITLE = "Chatbot"
PAGE_ICON = "💬"

# Backend switch
BACKEND = "hf"   # set to "ollama" to use Ollama again

# HF (PEFT) settings (used when BACKEND == "hf")
HF_LOAD_IN_4BIT = True   # requires bitsandbytes; hf_backend should use BitsAndBytesConfig
HF_MAX_NEW_TOKENS = 256  # ← fixed stray quote

# Llama access reminder (gated model)
if llama == 1:
    import os
    if not (os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")):
        print("[config] Note: Llama-3 base is gated. Set HUGGINGFACE_HUB_TOKEN with 'public gated repos' enabled.")
