# Supervised Fine-Tuning Roadmap: Core Concepts for Customizing LLMs

You now understand how language models learn — through next-token prediction, loss, and masking.

Now, before we dive into the actual implementation, we need to answer:
**What exactly are we fine-tuning, and how do all the upcoming topics fit together?**

This lesson lays out the journey through the foundational concepts you'll need before we fine-tune with LoRA and QLoRA.

We'll clarify what Supervised Fine-Tuning (SFT) really is, how it relates to training, and why topics like tokenization, quantization, and data types aren't random detours — they're essential building blocks.

By the end, you'll see the full map: where we're going, what each piece enables, and why understanding these foundations will transform you from someone who tweaks configs into someone who engineers solutions.

## What Is Supervised Fine-Tuning (SFT)?
Let’s start with the big picture.

**Supervised Fine-Tuning (SFT)** is the stage where a pretrained model learns to behave the way we want — to follow instructions, answer questions, or format responses in a specific style.

It uses the same learning mechanism you saw earlier — next-token prediction with cross-entropy loss — but now applied to structured, labeled examples that teach the model new behavior patterns.

In other words, fine-tuning doesn’t teach a model to *understand* language — it already does.
Fine-tuning teaches it **what kind of language to produce** in a given context.

### Example
A base model might continue this prompt:
*“Write an email to my boss explaining…”*
with something random or unhelpful.

SFT corrects that by showing it thousands of pairs like:
> **Instruction**: Write an email to your manager explaining you’ll miss today’s meeting.
> **Response**: Hi [Manager], I wanted to let you know I’m feeling unwell and won’t be able to attend today’s meeting. Thanks for understanding.

Over time, the model learns that given an instruction, a structured, context-aware response follows.
That’s how we turn a text completer into a capable assistant.

Under the hood, SFT is still just training continued — but with purpose.
You start from a pretrained model, use smaller, labeled datasets, and tune for specific tasks or styles instead of general language.

## What Makes Fine-Tuning Different from Pretraining?
Mechanically, SFT uses the same learning process as pretraining — next-token prediction with cross-entropy loss.

But there are three key differences:

1. **Starting Point**
   Pretraining starts from scratch (a model with randomized weights) whereas fine-tuning starts from a model that already understands language (a pretrained model).

2. **Data Structure and Scale**
   Pretraining uses raw, continuous text from massive datasets (trillions of tokens). Fine-tuning uses much smaller, structured pairs — instructions paired with desired responses (often just thousands to tens of thousands of examples).

3. **What Gets Trained On**
   In pretraining, the model learns from every token. In fine-tuning, we typically mask the instruction tokens — the model only updates its weights based on the assistant's response. This teaches it to produce appropriate outputs given specific inputs.

The underlying mechanism is identical. If you had unlimited data and compute, you could train from scratch for your task — we'd just call that "training" instead. These are labels based on starting conditions, not different algorithms.

## Where It Fits in the Training Journey
![Three Stages of LLMs](three-stages-of-lms.webp)

SFT sits in the middle of the modern LLM pipeline — between broad pretraining and preference-based alignment.

| Stage | Purpose | Data | Example Output |
| :--- | :--- | :--- | :--- |
| **1. Pretraining** | Learn general language and world knowledge | Web text, code, books | Base model (e.g., Llama 2) |
| **2. Supervised Fine-Tuning (SFT)** | Teach desired formats and response behaviors | Instruction–response pairs | Instruction-tuned model |
| **3. Preference Optimization** | Refine outputs to match human/model preferences | Ranked completions | Fully aligned assistant |

*Note: Preference optimization includes methods like DPO, GRPO, or PPO.*

We’ll focus on **Stage 2**, because SFT is the bridge that makes later steps possible.

Once a model learns to produce coherent, structured responses, techniques like Direct Preference Optimization (DPO), Generative Replay Optimization (GRPO) or Proximal Policy Optimization (PPO) can refine those responses even further based on feedback.

This program focuses on mastering that foundation — the supervised fine-tuning stage — because it’s the most accessible, reproducible, and scalable way to adapt large models for real-world use.

## What Comes Next: Core Concepts for Effective Fine-Tuning
Before we begin fine-tuning with LoRA and QLoRA, we need to understand the core concepts that make the process work.

![Core Concepts for Fine-Tuning](Core-Concepts-for-Effective-Fine-Tuning(4).png)

Over the next few lessons, we’ll cover:

| Concept | Purpose |
| :--- | :--- |
| **Dataset Preparation** | Structure and organize labeled examples for training. |
| **Tokenization and Padding** | Convert text to numeric sequences and align them for batching. |
| **Assistant-Only Masking** | Control which tokens contribute to loss to guide learning. |
| **Data Types and Quantization** | Manage memory/speed using FP16, BF16, FP8, or 4-bit/8-bit models. |
| **Parameter-Efficient FT (PEFT)** | Fine-tune efficiently by updating only a small subset of parameters (LoRA/QLoRA). |

Each of these topics is essential to fine-tuning responsibly and efficiently — ensuring your model trains faster, fits your hardware, and produces more stable results.

Think of them as the core engineering foundations that turn theoretical fine-tuning into a practical workflow. Once you understand these, fine-tuning stops being trial-and-error and becomes an engineering discipline.
