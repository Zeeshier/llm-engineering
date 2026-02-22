# Fine-Tuning GPT Models: OpenAI's Managed Fine-Tuning API Walkthrough

You’ve already built a baseline with Llama 3.2 1B. In the next lesson, you’ll fine-tune it yourself using LoRA in Colab.

Before that, let’s see what happens when we use a frontier model like **GPT-4o-mini** for the same SAMSum summarization task — first as-is, then after fine-tuning.

This gives us a clear comparison between open-weight and frontier models on the same dataset and shows how much **managed fine-tuning** can still improve a strong base model.

## Why This Managed Detour?
Managed services like OpenAI handle the infrastructure, optimization, and scaling for you. You upload data, they provide a tuned endpoint.

**Trade-offs to consider**:
* **Less Control**: No access to model internals or intermediate checkpoints.
* **Higher Cost**: Pay-per-token for training and inference.
* **Ease of Use**: No GPU management or CUDA troubleshooting required.

🎥 **Video Walkthrough: Fine-Tuning GPT-4o-mini**
See the complete workflow from JSONL preparation to ROUGE evaluation.

---

## Workflow 1: Evaluating the Frontier Baseline
We start by asking: *"How good is GPT-4o-mini out of the box?"*

### Mechanics: Messaging Format
OpenAI uses a chat-based message structure. For SAMSum, we roleplay as a helpful summarization assistant.
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant who writes concise, factual summaries."},
    {"role": "user", "content": f"Summarize the following conversation...\n\nDialog: {dialogue}"}
]
```

### Implementing Threaded Inference
Since we're calling a remote API, we use `ThreadPoolExecutor` to speed up evaluation across hundreds of samples.
```python
import concurrent.futures
import time

def generate_openai_predictions(model_name, dataset, task_instruction):
    # Uses max_workers to run concurrent API calls...
    pass
```

### Baseline Results
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 1B (Base)** | 35.12% | 13.04% | 27.26% |
| **GPT-4o-mini (Base)** | **39.97%** | **15.50%** | **31.28%** |

---

## Workflow 2: Managed Fine-Tuning
Now we fine-tune GPT-4o-mini using the exact same SAMSum dataset used for Llama.

### Step 1: Data Preparation (JSONL)
OpenAI expects data in a specific JSONL format.
```python
def convert_to_jsonl(dataset, out_path):
    with open(out_path, "w") as f:
        for sample in dataset:
            record = {
                "messages": [
                    {"role": "user", "content": build_prompt(sample["dialogue"])},
                    {"role": "assistant", "content": sample["summary"]}
                ]
            }
            f.write(json.dumps(record) + "\n")
```

### Step 2: Upload and Launch
```python
# Upload
train_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")

# Launch Job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=train_file.id,
    suffix="samsum-ft"
)
```

### Step 3: Monitoring & Evaluation
Once the status moves to `succeeded`, we use the new `tuned_model_id` in our existing evaluation loop.

---

## Final Comparison: The Performance Gap
| Model | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** | Open Base | 35.12% | 13.04% | 27.26% |
| **GPT-4o-mini** | Frontier Base | 39.97% | 15.50% | 31.28% |
| **GPT-4o-mini (Tuned)** | **Managed FT** | **55.72%** | **32.80%** | **47.77%** |

### Key Takeaways
1. **Scale Matters**: Frontier models start at a higher baseline.
2. **Fine-Tuning is the Multiplier**: Even the best models improve significantly when specialized on domain data (+16% ROUGE-L).
3. **Workflow Consistency**: We used the same prompt and metric across both models to ensure a fair "apples-to-apples" comparison.

Next, we return to the **self-hosted workflow** to achieve similar gains on open-weight models using QLoRA.
