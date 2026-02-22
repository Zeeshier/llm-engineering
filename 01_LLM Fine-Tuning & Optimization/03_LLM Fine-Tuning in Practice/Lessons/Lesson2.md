# SAMSum Fine-Tuning Project: Establishing Your Baseline Performance

In this lesson, you’ll meet the dataset we’ll be using this week — **SAMSum**, a benchmark for dialogue summarization — and see how well a base model like **Llama 3.2 1B Instruct** performs before any fine-tuning.

You’ll load the dataset, run a small evaluation, and record your model’s initial ROUGE score. That number becomes your reference point — the “before” snapshot that every fine-tuning experiment builds on.

By the end, you’ll understand the task, the metric, and the performance gap we’re aiming to close — the foundation for everything that follows in your first complete fine-tuning project.

## Your First Fine-Tuning Project: SAMSum
Each experiment starts by asking: *"What's our baseline performance — what are we trying to beat?"*

The **SAMSum** dataset contains real-world dialogues paired with human summaries.
> **Dialogue**:
> Amanda: I baked cookies. Do you want some?
> Jerry: Sure!
> Amanda: I'll bring you tomorrow :-)
>
> **Summary**:
> Amanda baked cookies and will bring Jerry some tomorrow.

### Why SAMSum?
* **Measurable**: Uses the ROUGE metric.
* **Efficient**: Fast enough for prototyping in Colab.
* **Realistic**: Mirrors actual chat/support summarization needs.

## Choosing the Base Model
We’ll use **Llama 3.2 1B Instruct**. 
* **Lightweight**: Runs easily in Colab.
* **Instruction-Tuned**: Understands tasks out-of-the-box.
* **Scalable**: Lessons learned here apply directly to larger 8B or 70B models.

## Running the Baseline Evaluation
In this section, we load the dataset and measure the "before" snapshot using ROUGE scores.

🎥 **Video Walkthrough: Baseline Evaluation**
Watch the full process of running Llama 3.2 1B on SAMSum to measure initial performance.

### Prerequisites (Dependencies)
```bash
pip install -q transformers datasets evaluate accelerate peft bitsandbytes
```

---

## The Evaluation Workflow

### Step 1: Configuration
We use a `config.yaml` to ensure reproducibility:
```yaml
dataset:
  name: knkarthick/samsum
  splits:
    train: 200
    validation: 200
    test: 200
```

### Step 2: Load and Prepare
We sample exactly 200 samples for a quick baseline test:
```python
def load_and_prepare_dataset(cfg):
    dataset = load_dataset(cfg["dataset"]["name"])
    # Sampling logic for train/val/test...
    return train, val, test
```

### Step 3: Setup Model
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16
)
# pad_token setup...
```

### Step 4: Generate Predictions
We prompt the model: *"Summarize the following conversation into a single sentence."*
```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Batch generation logic...
```

---

## What is ROUGE?
**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between model and reference summaries.

* **ROUGE-1**: Individual word overlap (Unigrams).
* **ROUGE-2**: Two-word sequence overlap (Bigrams).
* **ROUGE-L**: Longest Common Subsequence (Sentence structure).

## Baseline Results
The Llama 3.2 1B Instruct baseline (no fine-tuning) achieved:

| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | 35.1% |
| **ROUGE-2** | 13.0% |
| **ROUGE-L** | 27.2% |

Over the next lessons, we’ll use fine-tuning to push these numbers higher.

🎥 **Video: Summarization in Practice**
Learn why summarization is a top LLM application and what defines a "good" summary in the real world.
