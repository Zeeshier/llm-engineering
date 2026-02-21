# Instruction Fine-Tuning in LLMs: Assistant-Only Masking Explained

In previous lessons, you learned how language models work through next-token prediction and how text becomes model-ready through tokenization, padding, and attention masks.

Now it's time to solve a critical challenge: **making the model learn from only the right parts of the conversation.**

In instruction-following tasks, training data contains both user prompts and assistant responses — but we only want the model to learn to generate the assistant's part.

This selective learning is achieved through **assistant-only masking** — a technique that tells the model which tokens to learn from and which to ignore.

In this lesson, you’ll learn how masking works, why it matters, and how to verify it's working correctly in your fine-tuning pipeline.

## How Language Models Learn: A Quick Refresher
Language models learn by predicting the next token, one position at a time.

Given a sequence like *"The cat sat on the mat,"* the model tries to predict:
* Given "The" → predict "cat"
* Given "The cat" → predict "sat"
* Given "The cat sat" → predict "on"

Each prediction generates a **loss value** — how wrong was the prediction? — and that loss guides learning through backpropagation.

By default, the model computes loss at every position. But during instruction fine-tuning, that's not what we want.

## The Selective Learning Challenge
Each training example contains both user prompts and assistant responses, but we only want the model to learn to generate the **assistant's part**.

Consider this simple exchange:
> **User**: What's 2+2?
> **Assistant**: 4

If we compute loss at every position, the model will learn to predict every token — including "What's," "2+2?," and "Assistant:."

### Why is this a problem?
During inference, the model will never generate user input. The user provides that. The model only needs to generate the assistant's response. Training it to predict user tokens teaches the wrong behavior.

This is why base models (before instruction tuning) often mimic prompts or continue questions with more questions — they never learned to distinguish between reading a prompt and answering one.

**The solution**: Tell the model which tokens to learn from and which to ignore. That's what masking does.

## Assistant-Only Masking: Teaching the Model When to Learn
Masking solves this problem by controlling which positions contribute to the loss. In PyTorch, ignored tokens are marked with a special value, typically **-100**. When the loss function encounters -100, it skips that token entirely.

Imagine the same conversation, represented as tokens:
`[User, :, What's, 2+2?, Assistant, :, 4]`

We apply a mask so the model ignores everything before the assistant’s answer:
`[-100, -100, -100, -100, -100, -100, 4]`

Now, during training, loss is computed only on the final token — the assistant’s response. The model still sees the full context (user question included), but it only learns from predicting the assistant’s part.

## Comparing Training with and without Masking

### 1. Without Masking (Wrong Task)
```text
User: What's 2+2? Assistant: 4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Loss computed on ALL tokens
```
The model learns to predict user tokens, which it will never generate at inference.

### 2. With Masking (Correct Task)
```text
User: What's 2+2? Assistant: 4
                             ^
                   Loss only on assistant tokens
```
Now training aligns with inference: the model learns to generate assistant responses given conversation context.

🎥 **Video Walkthrough: Visualizing Assistant-Only Masking**
In this video, you’ll see how assistant-only masking works in practice using the Llama 3 Instruct tokenizer. We’ll visualize how -100 ensures only answer tokens contribute to loss.

## Masking Across Conversations
In multi-turn dialogues, user and system messages are masked, while each assistant turn remains unmasked.

### Concrete Example:
* **System**: You are a math tutor. `(masked)`
* **User**: What's 2+2? `(masked)`
* **Assistant**: It's 4. **(LEARNS)**
* **User**: And 3+3? `(masked)`
* **Assistant**: That's 6. **(LEARNS)**

The model sees the full conversation but only updates its weights based on the assistant's responses.

## How Masking is Implemented
Most of the time, you don’t need to handle masking manually. Modern frameworks like Hugging Face TRL and Axolotl handle it automatically through **chat templates**.

```python
formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
```

Once your data follows this format, the training framework applies assistant-only masking under the hood.

### Manual Implementation (Rare)
In custom setups, you might apply masking manually:
```python
def apply_assistant_masking(input_ids, labels, assistant_start_token):
    masked = labels.clone()
    start = (input_ids == assistant_start_token).nonzero(as_tuple=True)[0]
    if len(start) > 0:
        masked[:start[0]] = -100  # Mask everything before the assistant's response
    return masked
```

## Debugging Masking Issues
Masking errors are common. Here's how to spot them:

| Symptom | Diagnosis | Potential Fix |
| :--- | :--- | :--- |
| **Model echoes user input** | User turns aren't masked. | Verify chat template roles. |
| **Loss doesn't decrease** | Assistant turns might be masked. | Check labels aren't all -100. |
| **Generates system prompts** | System messages aren't masked. | Ensure system role is set to -100 in labels. |

## Wrap-Up
Assistant-only masking ensures the model learns exactly the behavior you want — generating helpful assistant responses rather than echoing user inputs. It aligns the training objective with the inference reality.
