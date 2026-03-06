<div align="center">

<img width="1920" height="1021" alt="Med-Qwen2.5-0.5B-MedReason-v2" src="https://github.com/user-attachments/assets/12d605bb-17a3-4e5b-bba5-df6f518a7b6c" />

<br/>
<br/>

# Med-Qwen2.5-0.5B-MedReason-v2

**A compact, high-performance medical reasoning model fine-tuned on 225,179 clinical reasoning pairs.**  
Structured differential diagnosis and mechanistic clinical reasoning at 500M parameters.

<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/HuggingFace-Rumiii%2FMed--Qwen2.5-ff9d00?logo=huggingface&logoColor=white)](https://huggingface.co/Rumiii/Med-Qwen2.5-0.5B-MedReason-v2)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--0.5B--Instruct-8b5cf6)](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit)
[![Framework](https://img.shields.io/badge/Framework-Unsloth-22c55e)](https://github.com/unslothai/unsloth)
[![Parameters](https://img.shields.io/badge/Parameters-500M-0ea5e9)]()
[![Dataset](https://img.shields.io/badge/Dataset-225%2C179%20Pairs-f43f5e)]()

</div>

---

## Overview

Med-Qwen2.5-0.5B-MedReason-v2 is a domain-specific language model fine-tuned for structured medical reasoning. Built on Alibaba's `Qwen2.5-0.5B-Instruct` and trained using LoRA via [Unsloth](https://github.com/unslothai/unsloth), the model produces numbered differential diagnoses, mechanistic clinical explanations, and evidence-grounded responses — competitive with models an order of magnitude larger.

The entire training pipeline ran on a single NVIDIA T4 GPU on Kaggle, demonstrating that high-quality medical reasoning capability can be instilled in sub-billion parameter models given the right data and fine-tuning methodology.

---

## Model Specifications

| Property | Value |
|:---|:---|
| Base Model | `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` |
| Parameters | 500 Million |
| Fine-Tuning Method | LoRA (PEFT) via Unsloth |
| LoRA Rank | r = 16 |
| LoRA Alpha | 16 |
| LoRA Target Modules | q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj |
| Training Steps | 1,000 |
| Max Sequence Length | 2,048 tokens |
| Quantization (Training) | 4-bit NF4 |
| Saved Format | Merged 16-bit |
| Training Hardware | NVIDIA T4 (Kaggle) |

---

## Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Optimizer | AdamW 8-bit |
| Learning Rate | 2e-4 |
| LR Scheduler | Linear decay |
| Batch Size (per device) | 2 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 16 |
| Warmup Steps | 5 |
| Weight Decay | 0.01 |
| Precision | bf16 / fp16 (hardware dependent) |
| Gradient Checkpointing | Unsloth optimized |
| Sequence Packing | Disabled |
| Seed | 3407 |

The model was saved using `save_pretrained_merged` with `merged_16bit` — producing a clean standalone checkpoint with no adapter overhead at inference time.

---

## Dataset

Training data was sourced from [OpenMed/Medical-Reasoning-SFT-Baichuan-M3-235B](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-Baichuan-M3-235B).

| Property | Detail |
|:---|:---|
| Source | OpenMed / Baichuan-M3-235B |
| Split Used | `train` |
| Total Rows | 225,179 |
| Format | Multi-turn chat (JSONL) |
| Template | Qwen 2.5 chat template |
| Coverage | Differential diagnosis, pharmacology, pathophysiology, evidence-based medicine |

**Preprocessing pipeline:**

1. Loaded full dataset from Hugging Face Hub
2. Stripped `reasoning_content` fields — only final answer `content` retained per message
3. Reformatted into Qwen 2.5 chat template via `get_chat_template`
4. Saved as JSONL and batched-mapped with `formatting_prompts_func`

---

## Repository Structure

```
Med-Qwen2.5-0.5B-MedReason-v2/
├── dataset_prep.py          # Dataset download, transformation, and JSONL export
├── train.py                 # Full LoRA fine-tuning script via Unsloth + SFTTrainer
└── app.py                   # Gradio chat interface for local and shared inference
```

---

## Quickstart

### Installation

```bash
pip install unsloth transformers torch
```

### Inference

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Rumiii/Med-Qwen2.5-0.5B-MedReason-v2",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{
    "role": "user",
    "content": "A 45-year-old male presents with sudden onset chest pain radiating "
               "to the jaw, diaphoresis, and shortness of breath. What is the most "
               "likely diagnosis and your reasoning?"
}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=1024,
    temperature=0.1,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)
```

### Running the Gradio Interface

```bash
python app.py
```

This launches a ChatGPT-style interface on a public share link via Gradio. The temperature slider controls response precision versus exploratory reasoning.

---

## Example Output

**Input**

> Is CTNNB1 45F mutation a molecular prognosticator of increased postoperative primary desmoid tumor recurrence — an independent, multicenter validation study?

**Output**

> Based on the provided information, yes, the CTNNB1 45F mutation appears to be a molecular prognosticator of increased postoperative primary desmoid tumor recurrence (DTR). The study found that patients with this mutation had significantly higher rates of postoperative DTR compared to those without it. The multicenter design provides robust external validation generalizable across institutions. The mutation's disruption of CTNNB1 — a protein involved in cell adhesion and migration — likely contributes to the desmoid recurrence phenotype. Results should be interpreted within the context of this specific patient population.

---

## Framework Dependencies

| Library | Role |
|:---|:---|
| [Unsloth](https://github.com/unslothai/unsloth) | Optimized LoRA fine-tuning and 4-bit inference |
| [TRL / SFTTrainer](https://github.com/huggingface/trl) | Supervised fine-tuning training loop |
| [Transformers](https://github.com/huggingface/transformers) | TrainingArguments and tokenization |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter management |
| [Datasets](https://github.com/huggingface/datasets) | Dataset loading and batched mapping |
| [Gradio](https://github.com/gradio-app/gradio) | Interactive inference interface |

---

## Limitations

- Trained for 1,000 steps on a single T4 — extended training on larger compute would improve coverage of rare conditions and edge cases
- Not evaluated against formal medical benchmarks (MedQA, PubMedQA, MedMCQA) in this release
- Not intended for clinical deployment or real-world patient care decisions
- May reflect distribution biases present in the source dataset

---

## Intended Use

This model is intended for medical NLP research and benchmarking, clinical reasoning dataset generation, educational demonstrations of efficient domain-specific fine-tuning, and prototyping medical AI interfaces.

It is **not** intended for direct clinical use, diagnostic support in live patient care settings, or any application requiring regulatory approval.

---

## Citation

```bibtex
@misc{med-qwen-medreason-v2,
  title   = {Med-Qwen2.5-0.5B-MedReason-v2},
  author  = {Rumiii},
  year    = {2025},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/Rumiii/Med-Qwen2.5-0.5B-MedReason-v2}
}
```

---

## Author

**Rumiii** — [Hugging Face](https://huggingface.co/Rumiii) · [GitHub](https://github.com/sufirumii/Med-Qwen2.5-0.5B-MedReason-v2)

---

<div align="center">
<sub>Fine-tuned with Unsloth on a single T4 GPU. Merged and released as a standalone 16-bit checkpoint.</sub>
</div>
