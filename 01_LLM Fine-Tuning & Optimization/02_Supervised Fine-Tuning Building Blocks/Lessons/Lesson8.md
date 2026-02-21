# Quantization and Double Quantization: How to Compress LLMs Efficiently

In the last lesson, you learned about data types—FP32, BF16, INT8, INT4, and NF4. You now know that a 7B model in FP32 needs about 8× more memory than in 4-bit precision.

But here's the practical question: **if you already have a model in FP32, can you actually convert it to 4-bit to save memory?**

Yes — and that's **quantization**. It's the process that takes trained models and compresses them from high-precision floats to compact integers. A 7B model that needs 28 GB in FP32 can fit in under 4 GB with 4-bit quantization. That's the difference between "needs a powerful cloud machine" and "runs on my laptop GPU."

In this lesson, you'll learn how quantization works: scale factors and zero points, why block-based quantization solves the outlier problem, and how double quantization compresses the metadata itself for an extra 10%-15% savings.

## Why Quantization Matters
Today, even storing all parameters of a large model can exhaust GPU memory before a single training step runs. Quantization fixes that by taking a radical but simple idea: you don’t always need 32-bit precision to represent weights. Instead, you can compress them into smaller integer formats — like INT8 or INT4 — without meaningfully changing model behavior.

## The Core Idea: Mapping Floats to Integers
Imagine your model weights are continuous numbers between -2.5 and +3.0. Quantization maps that smooth range into a small set of discrete bins — like rounding numbers to the nearest step on a ladder.

This mapping is governed by two quantities:
* **Scale**: How wide each bin is.
* **Zero point**: Which integer corresponds to real zero.

### A Simple Quantization Example (INT8)
Suppose we have a small set of model weights:
`weights = [0.7, -1.4, 2.5, -0.8, 1.9, -1.0, 0.3, 2.1, -0.5, 0.0]`

**Step 1 – Find the Range**
* `min_val = -1.4`
* `max_val = 2.5`
We map this range `[-1.4, 2.5]` to the INT8 range `[-128, 127]`.

**Step 2 – Calculate Scale Factor and Zero Point**
* **Scale**:
  $$scale = \frac{\max - \min}{255} = \frac{2.5 - (-1.4)}{255} \approx 0.0153$$
* **Zero Point**:
  $$zero\_point = \text{round}(-128 - \frac{\min}{scale}) = \text{round}(-128 - \frac{-1.4}{0.0153}) \approx -36$$

**Step 3 – Quantize**
Formula: `quantized = round(weight / scale + zero_point)`
* `0.7 / 0.0153 + (-36) \approx 10`
* `-1.4 / 0.0153 + (-36) = -128`
* `2.5 / 0.0153 + (-36) \approx 127`

**Step 4 – Dequantize (The Result)**
Formula: `dequantized = (quantized - zero_point) * scale`
* `(10 - (-36)) * 0.0153 \approx 0.704` (Original: 0.7)
* `(-128 - (-36)) * 0.0153 \approx -1.407` (Original: -1.4)

Precision loss is minimal, and neural networks are robust to these small errors.

### Memory Savings
| Configuration | Calculation | Size |
| :--- | :--- | :--- |
| **Original (FP32)** | 10 weights × 4 bytes | 40 bytes |
| **Quantized (INT8)** | 10 weights × 1 byte + 8b (scale/ZP) | 18 bytes (~2.2x smaller) |

In a real model with 1 million weights, the 8-byte overhead is negligible, resulting in a clean **4x reduction**.

---

## The Outlier Problem: Blockwise Quantization
If you use one global scale for an entire layer, a single large value (outlier) stretches the range, squashing all smaller values into the same bins and losing precision.

**The solution**: **Blockwise Quantization**. Divide the tensor into smaller chunks (e.g., 64 or 128 values). Each block gets its own scale and zero point. This ensures outliers in one part don't affect precision in another.

## Double Quantization: Compressing the Metadata
Each block needs its own metadata (scale/ZP). For a 7B model with a block size of 64, that's 109M blocks. Storing these as FP32 would take ~0.9 GB.

**Double Quantization** applies a second pass, quantizing those FP32 scale factors and zero points into 8-bit integers.

### 7B Model Memory Comparison
| Strategy | Weight Storage | Metadata Storage | Total Size |
| :--- | :--- | :--- | :--- |
| **Base Model (BF16)** | 14 GB | 0 | **14 GB** |
| **Quantized (4-bit)** | 3.5 GB | ~872 MB (FP32) | **~4.37 GB** |
| **Double Quant (4-bit)** | 3.5 GB | ~218 MB (INT8) | **~3.72 GB** |

*Double quantization saves an additional ~650 MB (15%) for a 7B model.*

---

## NF4 (NormalFloat4)
Neural network weights usually follow a Normal (Gaussian) distribution, clustering around zero. Standard INT4 spaces values uniformly, wasting precision in the sparse "tails."

**NF4** uses non-uniform spacing, placing more bins near zero where most weights live.
* **Result**: Quality loss drops from ~3-5% (Standard INT4) to ~1-2% (NF4) with zero memory cost.

---

## Implementation with BitsAndBytes
You can enable these optimizations automatically in Hugging Face:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit blockwise
    bnb_4bit_quant_type="nf4",              # Use optimized NormalFloat4
    bnb_4bit_use_double_quant=True,         # Compress metadata
    bnb_4bit_compute_dtype=torch.bfloat16   # Intermediate computation dtype
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Summary
Quantization is the bridge that makes fine-tuning massive models like Llama-3 70B possible on consumer hardware.
* **Blockwise Quantization** handles outliers.
* **Double Quantization** handles metadata overhead.
* **NF4** provides the most accurate 4-bit representation for weights.
Together, these enable up to **8x compression** with minimal quality loss.
