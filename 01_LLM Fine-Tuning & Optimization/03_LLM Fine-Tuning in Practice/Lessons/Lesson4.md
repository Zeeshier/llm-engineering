# Fine-Tuning Llama 3: Complete QLoRA Training Pipeline

You've built your baseline and evaluated GPT-4o-mini. Now it's time to take control — to fine-tune an open-weight model yourself.

In this lesson, you'll train **Llama 3.2 1B-Instruct** on the SAMSum dialogue summarization dataset using **QLoRA** — a combination of 4-bit quantization and LoRA adapters.

By the end, you'll have a reproducible training script, a set of LoRA adapters, and measurable performance gains over your baseline.

## From Baselines to Training
This workflow is implemented in modular Python scripts designed to scale. You'll reuse your earlier utilities for configuration and dataset loading, but focus on the core training logic:

1. **Assistant-only masking** to focus training on summaries.
2. **Trainer API configuration** for memory-efficient QLoRA.
3. **Hyperparameter management** via `config.yaml`.
4. **Adapter saving & evaluation**.

---

## Configuration: Your Training Blueprint
Everything is driven by a `config.yaml` to ensure reproducibility.

### 1. Quantization (for QLoRA)
```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: bfloat16
```
> [!NOTE]
> This reduces a 1B model's memory footprint from ~4GB to ~1GB, making it possible to train on almost any modern GPU.

### 2. LoRA Settings
```yaml
lora_r: 8
lora_alpha: 16
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```
> [!IMPORTANT]
> By targeting these modules, we train only ~2 million parameters instead of the full 1 billion.

---

## Fine-Tuning Implementation

### 1️⃣ Assistant-Only Masking
We compute loss **only** on the assistant's response (the summary). This preserves model capacity and prevents it from learning to predict the user's input.

```python
def preprocess_samples(examples, tokenizer, task_instruction, max_length):
    # Offset mapping identifies which tokens are "assistant" text
    # Mask everything else with -100
    labels = [-100] * start_idx + tokens["input_ids"][start_idx:]
    return {"input_ids": tokens["input_ids"], "labels": labels}
```

### 2️⃣ Training Configuration (Trainer API)
We use **Paged AdamW 8-bit** and **BFloat16** to keep memory usage low.

```python
args = TrainingArguments(
    output_dir="./outputs/lora_samsum",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()
```

### 3️⃣ Saving LoRA Adapters
LoRA adapters are tiny (~50MB). We save them separately from the frozen base model.
```python
model.save_pretrained("./outputs/lora_samsum/lora_adapters")
```

---

## Evaluating Results
To evaluate, we load the base model in 4-bit and then **attach** the adapters.

```python
model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb_config)
model = PeftModel.from_pretrained(model, "./lora_adapters")
```

### Final Comparison (SAMSum Benchmark)
| Model | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** | Open Baseline | 35.12% | 13.04% | 27.26% |
| **GPT-4o-mini (Base)** | Frontier Base | 39.97% | 15.50% | 31.28% |
| **Llama 3.2 1B (QLoRA)** | **Self-hosted FT**| **47.33%** | **22.17%** | **39.13%** |

### Key Takeaway
Your fine-tuned **Llama 3.2 1B now surpasses the base GPT-4o-mini** across all ROUGE metrics, even though it's much smaller. This shows the power of domain specialization via QLoRA.

## Summary
* You've built a complete, end-to-end pipeline.
* You've applied parameter-efficient training techniques.
* You've achieved measurable improvements over a much larger frontier model.

Next, we'll scale this up to **cloud GPUs (RunPod)** and add **experiment tracking (W&B)**.
