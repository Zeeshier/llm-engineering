# LoRA and QLoRA: Parameter-Efficient Fine-Tuning for LLMs

You've learned the building blocks of fine-tuning — how language models learn, how to prepare datasets, tokenization, padding, and instruction masking.

Now comes the practical breakthrough: **how do you actually fine-tune a 7B or 70B parameter model without a data center budget?**

This lesson introduces **LoRA** and **QLoRA** — the techniques that made large-scale fine-tuning accessible. You'll learn the mechanisms behind both approaches, when to use each one, and how to configure them for your hardware constraints.

## The Accessibility Problem
Traditional fine-tuning requires updating **every** parameter. For a 70B model, just storing weights in 16-bit precision takes ~140GB of VRAM. Add gradients and optimizer states, and you need over 300GB. This hardware is out of reach for most teams.

**Parameter-Efficient Fine-Tuning (PEFT)** changed everything. PEFT lets you fine-tune massive models by freezing the original weights and adding small trainable "adapters." Instead of billions, you train as few as 0.1% of the total parameters.

🎥 **Thinking Like a Problem Solver: PEFT Intuition**
Develop problem-solving intuition by approaching PEFT from first principles.

---

## Introducing LoRA: Low-Rank Adaptation
Introduced by Microsoft in 2021, LoRA fundamentally changed fine-tuning. Instead of modifying the original weight matrix $W$ (frozen), it learns an update matrix $\Delta W$ parameterized as the product of two smaller matrices, $A$ and $B$.

$$\Delta W = A \times B$$

Where $A$ is $m \times r$ and $B$ is $r \times n$. Since the rank $r$ is much smaller than the matrix dimensions, the total trainable parameters drop drastically.

### Visualizing the Impact
If $W$ is $5000 \times 4000$ (20 million parameters):
* **Full Fine-Tuning**: 20,000,000 trainable params.
* **LoRA (r=8)**: $A (5000 \times 8) + B (8 \times 4000) = 40,000 + 32,000 = 72,000$ trainable params.
* **Reduction**: **99.6% fewer parameters.**

🎥 **How LoRA Reduces Trainable Parameters**
Break down how LoRA decomposes weights layer by layer in a real LLaMA 3 model.

---

## LoRA Hyperparameters: The “Knobs” You Control
| Hyperparameter | Purpose | Rule of Thumb |
| :--- | :--- | :--- |
| **Rank (r)** | Adaptation capacity. | **8** is a good starting default. |
| **Alpha ($\alpha$)** | Scaling factor / "Volume". | **$\alpha = 2 \times r$** (e.g., 16). |
| **Target Modules** | Which layers to adapt. | **q_proj, v_proj** (Attention). |
| **Dropout** | Regularization. | **0.05 – 0.1** to prevent overfitting. |

> [!TIP]
> **Rank (r)**: Higher capacity (64+) uses more memory. Lower (4-8) is faster.
> **Targeting Modules**: Spread moderate ranks across more modules (e.g., all linear layers) often works better than a very high rank on just one.

🎥 **Configuring LoRA Hyperparameters**
Learn how to set rank, alpha, and target modules based on memory and objectives.

---

## QLoRA: Quantized Low-Rank Adaptation
QLoRA (2023) combined quantization with LoRA to enable fine-tuning even larger models on consumer GPUs.

1. **NF4 (NormalFloat-4)**: The optimal 4-bit representation for normally distributed weight data.
2. **Double Quantization**: Compresses the quantization metadata itself, saving an extra ~15% memory.
3. **Paged Optimizers**: Moves optimizer states to CPU RAM during memory spikes, preventing "Out of Memory" (OOM) errors.

### Training Logic
Base weights stay in **4-bit** (frozen). During computation, they are used as 16-bit, but only the LoRA adapters (small fraction) accumulate gradients and get updated.

---

## LoRA vs QLoRA: When to Use Each?
| Feature | LoRA | QLoRA |
| :--- | :--- | :--- |
| **Memory** | High (16-bit) | **Low (4-bit)** |
| **Speed** | **Faster** | ~10-20% slower (due to dequant) |
| **Hardware** | Enterprise GPUs (A100) | **Consumer GPUs (RTX 3090/4090)** |
| **Quality** | Baseline | Nearly identical to LoRA |

**Recommendation**:
* Fits in VRAM? → Use **LoRA**.
* Doesn't fit? → Use **QLoRA**.

---

## Implementation Examples

### LoRA (Standard)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### QLoRA (Quantized)
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("Llama-3-8B", quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

## Summary
* **LoRA** reduces trainable parameters by 10,000x using low-rank decomposition.
* **QLoRA** reduces weight storage by 4x-8x using quantization.
* Together, they allow you to fine-tune **70B parameter models** on hardware that fits on your desk.
