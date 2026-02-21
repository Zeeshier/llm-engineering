# Dataset Preparation for LLM Fine-Tuning: Formats and Best Practices

You now understand what Supervised Fine-Tuning (SFT) is — the stage where we teach models to follow instructions and produce structured, human-like responses.

But before we fine-tune anything, we need to prepare the one thing the model actually learns from: **data**.

This lesson bridges the gap between concept and implementation. You’ll learn how fine-tuning datasets are structured, where they come from, and how to create or adapt them for your own projects.

We’ll explore what makes a dataset instruction-ready — from formats like Alpaca and OASST, to multi-turn messages structures used in chat models — and how to validate, clean, and publish your datasets using the Hugging Face Datasets library.

By the end, you’ll know how to move from raw text or synthetic data to a clean, well-formatted dataset ready for tokenization and fine-tuning — the critical next step in your LLM workflow.

## Everything Starts with Data
When you train or fine-tune a large language model, three things shape its behavior:
1. **The dataset** — what the model sees and how that information is structured.
2. **The model architecture** — how it processes and represents that information.
3. **The loss function** — how it’s told what “good” behavior looks like.

All three matter — but if you had to pick one, **the dataset usually carries the most weight**. It defines what the model learns to imitate, generalize, and prioritize.

The architecture is mostly fixed — we’ll choose an existing model like Llama, Mistral, or Qwen and apply efficient tuning methods such as LoRA or QLoRA.
The loss function can vary — from standard cross-entropy with assistant-only masking to more advanced schemes like GRPO or DPO.
But the dataset is what truly drives the outcome. It encodes the behavior you want the model to exhibit.

That’s why dataset preparation is the first hands-on step in fine-tuning. Before we get into tokenization, masking, and parameter-efficient methods, we need to ensure the data — the model’s learning signal — is clean, structured, and aligned with the task.

In this lesson, we’ll focus entirely on that: where datasets come from, how they’re structured, and how to prepare them for Hugging Face fine-tuning workflows.

## Understanding Dataset Sources and Types
When you fine-tune a model, you’re not just picking any dataset — you’re defining what kind of behavior the model should learn. That starts with your task.

If you’re building a chatbot for a financial services company, your dataset should reflect that domain — either as instruction–response pairs for chat models, or as plain factual text for continued pretraining or knowledge adaptation. The goal is **alignment**: your data should look like the interactions or text your model will handle in production.

In real-world projects, datasets typically come from three places:
* **Internal data sources** — such as documents, FAQs, transcripts, or reports.
* **Public datasets** — for prototyping or baseline experiments (e.g., FLAN, OASST, Alpaca).
* **Custom data collection** — when neither of the above fully represents your use case.

Once you have the source, you can decide how to build or expand it. Broadly, there are three approaches:

### Human-Labeled Datasets
Still the gold standard. Human annotators create or verify examples to ensure clarity, consistency, and alignment. These datasets tend to be smaller but far more reliable — ideal for specialized assistants or high-stakes domains.

### Synthetic and LLM-Generated Datasets
The fastest-growing approach. You can use large models to generate structured examples from seed material — a method popularized by Self-Instruct and Alpaca. This is especially useful for domain-specific tasks, where you can guide generation with clear prompts and output schemas.

But synthetic data isn’t automatically high quality. It must be validated and diversified, or it risks producing narrow or biased model behavior.

### Hybrid Data Creation
Most modern fine-tuning projects mix both — generating examples with an LLM, then filtering or ranking them with human review or heuristic checks. This hybrid strategy balances scale and quality.

> [!IMPORTANT]
> A simple rule to remember: if your dataset doesn’t reflect the diversity and reality of the task, even the best model and loss function can’t save you. **Garbage in, garbage out.**

For the projects in this program, you can freely use public datasets to learn the mechanics of fine-tuning. But in real deployments, dataset creation is often the hardest — and most valuable — part of the process.

## Dataset Types by Training Stage
Each stage of the LLM lifecycle uses a different type of dataset:

| Type | Purpose | Example Datasets |
| :--- | :--- | :--- |
| **Pretraining** | Teach general language and world knowledge | The Pile, WikiText |
| **Supervised Fine-Tuning (SFT)** | Teach structured, instruction-following behavior | Alpaca, FLAN, OASST |
| **Preference / Reward Training** | Align model preferences or tone | ORPO, HH-RLHF |

Supervised Fine-Tuning — the focus of this lesson — sits in the middle. It’s where your dataset moves from unstructured text to structured examples that show the model how to behave.

## SFT and Preference / Reward Training Formats
Let’s look at the most common schemas for fine-tuning and preference / reward training.

### 1. Instruction Format
The simplest and most common schema for single-turn tasks. Each example includes an instruction, an optional input, and an output — exactly what the model should generate.

**Example (Alpaca-style):**
```json
{
  "instruction": "Translate to French",
  "input": "Hello, world!",
  "output": "Bonjour, le monde !"
}
```
This format works beautifully for summarization, translation, and Q&A. The model learns: when I see this kind of instruction, I produce this kind of output.

### 2. Conversation (Chat) Format
For multi-turn chat data, each record contains a list of messages — each labeled with a role and content.

**Example (OASST-style):**
```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is 2 + 2?" },
    { "role": "assistant", "content": "4" }
  ]
}
```
This structure mirrors the way we chat with assistants like ChatGPT or Claude. Training on this schema teaches the model conversational flow — when to respond, when to stop, and how to maintain context.

### 3. Preference Format
Used in later stages like DPO or ORPO, where models learn which answer is better. Each row includes two responses to the same prompt — one “chosen” and one “rejected”:

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses qubits that can represent 0 and 1 at once.",
  "rejected": "Quantum computers use large databases of classical bits."
}
```
While we won’t train on this format yet, it’s worth recognizing — many fine-tuning pipelines evolve into preference training later.

### 4. Pretraining Format: Plain Text
The pretraining phase is all about teaching the model general patterns of language. The dataset is typically a collection of plain text, with each row containing a single field called `text`.

```json
{
  "text": "The quick brown fox jumps over the lazy dog."
}
```

## Format Comparison Summary
| Format | Primary Use Case | Fields | Example Dataset |
| :--- | :--- | :--- | :--- |
| **Pretraining** | Base models | `text` | The Pile, WikiText |
| **Instruction** | Single-turn tasks | `instruction`, `input`, `output` | Alpaca, FLAN |
| **Conversation** | Multi-turn chat | `messages` (role, content) | OASST, ShareGPT |
| **Preference** | Alignment & reward | `prompt`, `chosen`, `rejected` | ORPO, HH-RLHF |

Consistency within one schema is key. Frameworks like Hugging Face TRL and Axolotl handle chat formatting automatically — as long as the roles and delimiters are consistent.

## Linearization and Chat Templates
The dataset structures we've discussed are for our convenience. When it's time to train, these fields are merged into a single text string according to a **predefined chat or instruction template**.

For example, an Alpaca entry might be rendered internally as:
```text
### Instruction:
Summarize the text below.

### Input:
Large language models are revolutionizing AI.

### Response:
LLMs are transforming artificial intelligence.
```

Frameworks like Hugging Face TRL or Axolotl handle this automatically:
```python
formatted = tokenizer.apply_chat_template(example, tokenize=False)
```
The template also adds special tokens like `<|begin_of_text|>` and `<|end_of_text|>` — we'll cover that in detail in the next lesson on tokenization.

## Creating, Validating, and Managing Datasets
Once you understand the formats, the next step is getting your data into one.

### Finding Datasets on Hugging Face
Hugging Face hosts thousands of fine-tuning datasets. You can browse them at [huggingface.co/datasets](https://huggingface.co/datasets) or load them directly in code:

```python
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")
print(dataset["train"][0])
```

🎥 **Video Walkthrough: Exploring Datasets on Hugging Face**
In this lesson, you’ll explore how the Alpaca dataset format structures data for instruction tuning. You’ll learn to load datasets from Hugging Face, understand how instruction, input, and output fields work, and see how the final text field combines everything into a training-ready format.

### Creating Your Own Dataset
You can create datasets manually, or more efficiently, with an LLM-assisted pipeline like **Distilabel**:

```python
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, TextGeneration

with Pipeline(name="qa-generator") as pipeline:
    load = LoadDataFromHub(output_mappings={"content": "instruction"})
    generate = TextGeneration(
        llm=OpenAILLM(model="gpt-4-turbo", api_key="YOUR_KEY"),
        input_batch_size=4
    )
    load.connect(generate)

dataset = pipeline.run(parameters={"repo_id": "my/seed-data"})
```

### Validating and Cleaning Data
Quality control is non-negotiable. A few checks can save hours of wasted GPU time:
1. Every row follows the expected schema.
2. No empty or truncated responses.
3. Instruction and output make sense together.
4. Duplicates and near-duplicates are removed.

```python
def clean(example):
    return example["output"].strip() != ""

dataset = dataset.filter(clean)
```

### Preparing and Publishing
Once your dataset is cleaned, you can save or publish it:

```python
# Convert and Save
dataset.save_to_disk("data/alpaca_clean")
dataset.to_json("data/alpaca_clean.json")

# Push to Hugging Face Hub
dataset.push_to_hub("username/alpaca-clean")
```

🎥 **Video: Pushing Datasets to Hugging Face Hub**
In this video, you’ll learn how to create, split, and upload your instruction-tuning datasets to Hugging Face Hub, so they’re easy to share and reuse during fine-tuning. We’ll cover authentication, dataset creation using the datasets library, and pushing data.
