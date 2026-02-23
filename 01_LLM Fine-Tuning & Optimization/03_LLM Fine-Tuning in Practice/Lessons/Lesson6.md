# Weights & Biases Tutorial: Tracking LLM Fine-Tuning Experiments

You just fine-tuned your first model. It worked — you got results, saw improvement, and pushed it to the Hub. But here's the question: *which hyperparameters did you use again?* Was that LoRA rank 8 or 16? Learning rate 1e-4 or 1e-5?

This is where **experiment tracking** comes in. As you start running multiple experiments, you need a system to track what you tried, what worked, and why.

In this lesson, we’ll integrate **Weights & Biases (W&B)** into your pipeline to bring structure to your experimentation.

## Why Experiment Tracking Matters
Without tracking, experiments become chaos — endless folders and lost metrics. With W&B, everything is automatically logged to the cloud. You can revisit every run later — parameters, loss curves, even system stats — keeping your work **reproducible**.

## Getting Started with W&B
Weights & Biases is the industry standard for LLM work. It provides:
* **Automatic Logging**: Integrates directly with the Hugging Face `Trainer`.
* **Visual Dashboards**: Compare metrics across runs in real-time.
* **Artifact Tracking**: Save model checkpoints and dataset versions alongside metrics.
* **System Stats**: Monitor GPU memory and utilization.

### Setup
1. Create a free account at [wandb.ai](https://wandb.ai).
2. Install the library:
```bash
pip install -q wandb
```
3. Authenticate:
```python
import wandb
wandb.login()
```

---

## Integrating W&B into Your Pipeline
Adding W&B to your fine-tuning script takes just two steps.

### Step 1: Initialize W&B
Call `wandb.init` at the start of your script to define the project and initial metadata.
```python
import wandb

wandb.init(
    project="samsum-fine-tuning",
    name="llama-1b-lora-r8-baseline",
    config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "lora_r": 8,
        "learning_rate": 2e-4
    }
)
```

### Step 2: Update TrainingArguments
Set `report_to="wandb"` in your `TrainingArguments`.
```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./outputs",
    learning_rate=2e-4,
    report_to="wandb",  # 👈 This enables the integration
    logging_steps=10
)
```
Now, whenever you call `trainer.train()`, Hugging Face will automatically pipe all metrics to your W&B dashboard.

---

## Planning Your Experiments
Now that you have tracking, you can run a **Hyperparameter Search**. Here are three key experiments we’ll run on RunPod:

### 1. LoRA Rank ($r$)
* **Test**: $r = 4, 8, 16, 32$
* **Goal**: Find the point of diminishing returns for adapter capacity.

### 2. Learning Rate
* **Test**: $1e-4$ vs $1e-5$
* **Goal**: Balance convergence speed with training stability.

### 3. Target Modules
* **Test**: `{q, v}` vs `{q, k, v, o}`
* **Goal**: Determine if targeting all attention projections justifies the extra memory usage.

## What to Watch in the Dashboard
* **Loss Curves**: Are they smoothing out, or is there instability?
* **GPU Memory**: How much headroom do you have left for larger batches?
* **ROUGE vs Steps**: How many steps are actually needed before improvement plateaus?

By the end of these experiments, you’ll have a data-driven configuration that delivers the best possible model for your task.
