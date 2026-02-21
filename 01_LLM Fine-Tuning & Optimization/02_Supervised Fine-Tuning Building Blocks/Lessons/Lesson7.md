# Data Types in Deep Learning: FP32, FP16, BF16, INT8, INT4 Explained

Every number inside a large language model — every weight, activation, and gradient — has a data type.

It might sound like a small detail, but the choice of data type determines how fast your model trains, how much memory it consumes, and even how stable it is.

In this lesson, you’ll unpack what formats like FP32, FP16, BF16, INT8, and INT4 really mean, how they trade off precision for performance, and why modern LLMs rarely stick to just one.

By the end, you’ll know exactly which data type to use for training, fine-tuning, or inference — and how precision decisions can make or break large-scale model performance.

## Why Data Types Matter
Every model you train — whether it’s a 1B parameter LLaMA variant or a 70B behemoth — is just a huge collection of numbers. Each of those numbers has a format: 32 bits, 16 bits, or even 4 bits.

This choice affects:
* **GPU Memory**: How many parameters can you fit?
* **Speed**: How quickly can specialized hardware (Tensor Cores) process the math?
* **Stability**: Can the model represent tiny gradients without them vanishing to zero?

## Floating-Point Data Types Explained
Floating-point numbers are built from three parts:
1. **Sign bit** — Positive or negative?
2. **Exponent** — How large or small is the number? (Controls the **range**)
3. **Mantissa** (or significand) — What are the actual digits? (Controls the **precision**)

The key trade-off: **Range vs. Precision.**

### Data Type Comparison
| Format | Bits | Sign | Exponent | Mantissa | Range | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 1 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | ~7-8 digits |
| **FP16** | 16 | 1 | 5 | 10 | $\pm 65,504$ | ~3-4 digits |
| **BF16** | 16 | 1 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | ~2-3 digits |

🎥 **Video Walkthrough: Understanding FP32, FP16, and BF16 in PyTorch**
Learn how floating-point data types affect precision and memory, and see live calculations comparing tensors in PyTorch.

---

### FP32 (Float32) — The Full-Precision Baseline
The traditional standard. It's stable and precise, but expensive.
* **Memory**: 4 bytes per parameter.
* **Problem**: A 7B parameter model takes 28 GB just for weights. With gradients and optimizer states, you might need 50+ GB for training.

### FP16 (Float16) — Half Precision
Cuts memory in half (2 bytes per parameter), but the narrow **range** causes issues.
* **Underflow/Overflow**: Gradients often exceed 65,504 or drop below the smallest representable number, leading to NaN or zero.
* **Fix**: Requires "Loss Scaling" during training to prevent instability.

### BF16 (BFloat16) — The Modern Standard
Designed by Google specifically for deep learning. It uses 16 bits but keeps the same 8-bit exponent as FP32.
* **Range**: Same as FP32 ($\pm 10^{38}$), so no loss scaling is needed.
* **Precision**: Less than FP16, but neural networks are robust to small rounding errors. **Stability is more important than precision.**
* **Support**: NVIDIA A100/H100, TPUs.

---

## Integer Data Types for Inference
When a model is trained, you can use integers to save more memory. This process is called **Quantization**.

### INT8 — Efficient and Reliable
Stores numbers as 8-bit integers (-128 to 127). Uses 1 byte per parameter.
* **Process**: Maps continuous float values to integers and scales them back during inference.

### INT4 — Extreme Compression
Uses only 4 bits per number (values from 0-15 or -8 to 7).
* **QLoRA**: Enables fine-tuning large models on consumer GPUs by compressing the base model to 4 bits while kept trainable adapters in higher precision.

### NF4 (NormalFloat-4) — Optimized for Neural Networks
Standard 4-bit formats spread values uniformly. However, weights usually cluster around zero. **NF4** places more resolution near zero, preserving model quality far better.
* **Use Case**: Default for QLoRA base model weights.

---

## Memory Requirements: Practical Calculations
The basic formula for weight storage:
> **Memory (Bytes) = Parameters × Bytes per Parameter**

| Model Size | FP32 (4B/p) | BF16/FP16 (2B/p) | INT8 (1B/p) | INT4 (0.5B/p) |
| :--- | :--- | :--- | :--- | :--- |
| **7B** | 28 GB | 14 GB | 7 GB | 3.5 GB |
| **13B** | 52 GB | 26 GB | 13 GB | 6.5 GB |
| **70B** | 280 GB | 140 GB | 70 GB | 35 GB |

> [!NOTE]
> During training, you need 3–4× this amount for gradients, activations, and optimizer states.

🎥 **Video: Calculating Model Memory Requirements**
Learn how to estimate GPU memory needs and see how switching from FP32 to INT4 reduces memory by up to 8×.

## Summary: Precision Cheat Sheet
| Phase | Recommended Precision | Reason |
| :--- | :--- | :--- |
| **Full Training** | **BF16** | Stable and efficient. |
| **Fine-Tuning** | **FP16 / BF16** | Balance of speed and precision. |
| **Inference** | **INT8 / INT4** | Small, fast, cost-efficient. |
