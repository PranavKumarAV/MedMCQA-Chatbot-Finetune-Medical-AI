# MedMCQA Chatbot – Fine‑tune · Medical AI

Fine‑tune, evaluate, and deploy **Qwen2.5‑7B‑Instruct** and **Meta‑Llama‑3‑8B‑Instruct** on **MedMCQA** (subjects: **Biochemistry** and **Physiology**).  
This app **auto‑downloads** the base model + **LoRA** adapter from Hugging Face and serves a medical MCQ tutor in Streamlit.  
> Educational use only. Not medical advice.

---

## What this repo includes
- Training notebooks/scripts (Unsloth + QLoRA) to produce **PEFT/LoRA** adapters.
- Streamlit app (`main.py`) with a simple toggle between **Qwen** and **Llama** backends.
- HF backend that **fetches models automatically** via `transformers` + `peft` (no manual copy/paste of weights).
- Answer‑only evaluation + runtime metrics for **model selection**.
- Model cards on HF (adapters) you can use directly.

**Subjects used:** **Biochemistry**, **Physiology** (subset to fit T4 16 GB; stratified 70/30 split on `subject_name`).

---

## Requirements (minimal)
- Python 3.10+
- CUDA‑capable GPU recommended (e.g., T4 16 GB)
- Key libs (see `requirements.txt` for exact pins): `transformers`, `peft`, `unsloth`, `bitsandbytes`, `streamlit`, `torch`

---

## Model selection (accuracy vs latency)

### Results (your measured runs)
| Model | Internal val acc (%) | Original val acc (%) | TTFT (ms) | Gen time (ms) | In/Out tokens |
|---|---:|---:|---:|---:|---:|
| Llama‑3‑8B (LoRA) | 83.83 | 65.20 | 567 | 14874 | 148 / 80 |
| Qwen2.5‑7B (LoRA) | 76.50 | 67.84 | 546 | 1623  | 81 / 15 |

**Interpretation (short):**
- **External accuracy:** Qwen higher (67.84 % vs 65.20 %).  
- **Latency:** Qwen ~9× faster total gen; lower ms/token.  
- **Internal split:** Llama higher (83.83 %).

**Decision:** Default to **Qwen** for **better generalization** and **interactive latency**. Keep **Llama** as an **optional “Explain mode”** (richer, longer answers).

**Speed tip for Llama (answer‑only):**
```python
# set smaller budget
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
# cut after first "Answer: X"
text = tok.decode(out[0], skip_special_tokens=True)
import re
m = re.search(r"Answer:\s*([A-D])\b", text)
final = f"Answer: {m.group(1)}" if m else text.strip()[:32]
```

---

## Quick start

```bash
# 1) Create/activate env
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Only if using Llama) Accept the base model license on HF
#    and create a fine‑grained token with "Access to public gated repos".

# 4) Run the app
streamlit run main.py
```

The app will **pull the base + LoRA from Hugging Face automatically**. No manual downloads required.

---

## Configuration (matches the code)

In `config.py` you select the backend model with a simple toggle:

```python
# To test different models
llama = 0  # 0 - Qwen2.5-7B-Instruct; 1 - Llama-3-8B-Instruct

if llama == 1:
    MODEL = "Llama-3-8B-Instruct"
    HF_BASE_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_ADAPTER_PATH = "Pk3112/medmcqa-lora-llama3-8b-instruct"
else:
    MODEL = "Qwen2.5-7B-Instruct"
    HF_BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
    HF_ADAPTER_PATH = "Pk3112/medmcqa-lora-qwen2.5-7b-instruct"

# App
MAX_TURNS = 20
BUDGET_CHARS = 12000
PAGE_TITLE = "Chatbot"
PAGE_ICON = "💬"

# Backend
BACKEND = "hf"           # use Transformers + PEFT
HF_LOAD_IN_4BIT = True   # BitsAndBytes 4‑bit
HF_MAX_NEW_TOKENS = 256
```

### Hugging Face access
- **Qwen path**: Apache‑2.0, **no gating**. Works without a token.  
- **Llama path**: **gated**. Do both:
  1) Accept the **Meta‑Llama‑3** license on its HF page.  
  2) Create a **fine‑grained token** with *“Access to public gated repos”* enabled and make it visible to the app, e.g.:
     - `huggingface-cli login`  
     - **Windows (persist):** `setx HUGGINGFACE_HUB_TOKEN <token>` then restart shell  
     - **Session only:** `$env:HUGGINGFACE_HUB_TOKEN="<token>"`

---

## How it works (inference)

- Loads the **base model** (`HF_BASE_ID`) with optional **4‑bit** via `BitsAndBytesConfig`.
- Loads the **LoRA adapter** (`HF_ADAPTER_PATH`) using `peft.PeftModel.from_pretrained(...)`.
- Streams tokens to the UI and records **TTFT**, **generation time**, and **tokens in/out**.
- For fair comparison, decoding uses `do_sample=False` (greedy). For Llama speed‑ups, cap `max_new_tokens` and stop after the letter.

---

## Training & evaluation (summary)

- **Frameworks:** Unsloth + PEFT/LoRA (QLoRA NF4).  
- **LoRA:** `r=32`, `alpha=64`, `dropout=0.0`; targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.  
- **Context:** `max_seq_length=768`.  
- **Objective:** **answer‑only** (`Answer: <A/B/C/D>`).  
- **Split:** train/val created by stratifying `subject_name`; subjects fixed to **Biochemistry** & **Physiology**.  
- **Eval:** decode‑free A/B/C/D scorer for accuracy; small explanation probe.  
- **Selection rubric:** maximize **external accuracy** subject to **latency** (TTFT + ms/token). Qwen chosen as default; Llama reserved for explain mode.

---

## Training code & reproducibility
- Training notebook/script is included in this repo.  
- GitHub (recommended to pin): add a release tag (e.g., `v1.0-medmcqa`) and record the commit SHA used for training. Link both from your HF model cards.

---

## Troubleshooting

- **403 on Llama base:** accept the license on HF and use a fine‑grained token with *public gated repos* enabled. Restart Streamlit after `huggingface-cli login` (or re‑open your terminal if you used `setx`).  
- **“adapter_config.json not found” or invalid repo id:** set `HF_ADAPTER_PATH` to a valid HF repo id (e.g., `Pk3112/medmcqa-lora-qwen2.5-7b-instruct`) or to a real local folder that contains `adapter_model.safetensors` + `adapter_config.json`.  
- **Empty stream:** the app auto‑falls back to non‑streaming; check console logs for exceptions.

---

## Safety

This is an educational tutor. Do **not** use for diagnosis or treatment. Always verify with authoritative medical sources.

---

## License

- App code: your repository license.  
- **Adapters:** hosted on HF under their model cards.  
- **Base models:** subject to the original licenses (Qwen: Apache‑2.0; Llama‑3: Meta Llama 3 Community License).
