# Tokenization and Padding: Preparing Text Data for LLM Training

Before you can fine-tune a model, you need to speak its language — **numbers**.

Language models don’t read sentences; they process **tokens** — small subword units that represent pieces of meaning.

But once you’ve turned text into tokens, another challenge appears: every sentence has a different length. Models train in batches, which means all sequences must line up. That’s where **padding** comes in — to make sequences uniform.

Once you introduce padding, though, you create a new problem: the model might learn from filler tokens instead of real content. That's why we need **attention masks** — they tell the model which tokens are real and which to ignore.

In this lesson, you’ll see how tokenization, padding, and attention masking work together to convert raw text into the structured numerical format that every large language model needs before it can learn anything.

## From Text to Tokens
Language models can’t read like humans — they interpret text through numbers. Each word, subword, or symbol is turned into a **token ID**, a unique integer that represents a fragment of meaning.

Before a model can learn from text, it must pass through a preprocessing pipeline:
1. **Tokenization** — break text into small, meaningful units (tokens).
2. **Conversion** — map tokens to numerical IDs.
3. **Padding and masking** — make sequences the same length for batch processing while telling the model what to ignore.

This transformation — from messy human text to ordered numeric tensors — is the foundation of every fine-tuning workflow.

## How Tokenization Works
The simplest way to split text might seem obvious: separate by spaces. But that approach fails when faced with contractions (“don’t”), compound terms (“state-of-the-art”), or new words like “ChatGPT.”

Tokenization solves this by breaking language into smaller, reusable pieces. Modern LLMs use **subword-level tokenization**.

| Approach | Example | Strength | Limitation |
| :--- | :--- | :--- | :--- |
| **Word-Level** | "The cat sat..." → ["The", "cat", "sat"] | Intuitive/Readable | Huge vocabulary, OOV errors |
| **Character-Level** | → ["T", "h", "e", " ", "c", ...] | Represents everything | Too granular/Inefficient |
| **Subword-Level** | → ["The", "▁cap", "ital", "▁of"] | Balanced/Handles new words | None significant today |

Subword tokenization hits the sweet spot. It keeps the vocabulary manageable (30k–100k tokens) while allowing unknown words to be expressed as combinations of familiar parts. For example, “unbelievable” becomes `["un", "believ", "able"]`.

### Subword Tokenization in Action
Let’s test it on something messy: *“ChatGPT-4.5-turbo-ultra-mega-awesome”*
* A word-level tokenizer fails.
* A character-level tokenizer expands it to 30+ symbols.
* A subword tokenizer breaks it down intelligently: `["Chat", "GPT", "-", "4", ".", "5", "-", "turbo", "-", "ultra", "-", "mega", "-", "awesome"]`.

### Video Walkthroughs
🎥 **Understanding Tokenization in LLMs**
Learn how tokenization transforms plain text into token IDs for models like LLaMA and Mistral.

🎥 **Special Tokens and Chat Templates**
Explore tokens like BOS, EOS, and PAD, and see how `apply_chat_template()` converts dialogue into model-ready input.

🎥 **How OpenAI Tokenization Works**
See how GPT models tokenize text using subword pieces and learn to use the `tiktoken` library.

## Implementing Tokenization with Hugging Face
Tokenization is straightforward with the `transformers` library:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
text = "What's the capital of France?"

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")
```

### Special Tokens
Every tokenizer defines special tokens that mark important boundaries:
* `bos_token`: Beginning of sequence (e.g., `<s>`)
* `eos_token`: End of sequence (e.g., `</s>`)
* `pad_token`: Padding (used to align batch lengths)
* `unk_token`: Unknown token

## Padding and Batching: Keeping Sequences Aligned
Different sentences produce different sequence lengths. Models process data in batches, so all examples in a batch must have the same shape.

**Padding** adds a special token (usually `<pad>`) to shorter sequences so they match the longest one.

**Example:**
* "Hi!" → `[1, 2345, 3]`
* "What's the weather like?" → `[1, 456, 78, 901, 234, 567, 8, 3]`

**After padding (to length 8):**
* "Hi!" → `[1, 2345, 3, 0, 0, 0, 0, 0]`

### Padding Strategies
1. **Right Padding (default)**: Add tokens to the end — ideal for most training tasks.
2. **Left Padding**: Add to the beginning — used during generation (autoregressive).

🎥 **Video: Padding and Attention Masks**
Learn how padding and attention masks enable batch processing in LLMs.

## Attention Masks: Telling the Model What to Ignore
Padding tokens shouldn't affect the model's learning. **Attention masks** are binary arrays marking real tokens as `1` and padding as `0`.

**Example:**
* **Input IDs**: `[1, 2345, 3, 0, 0]`
* **Attention Mask**: `[1, 1, 1, 0, 0]`

This ensures the model focuses on real content and ignores padding during loss calculation.

## Best Practices and Common Pitfalls
✅ **Consistency**: Use the same tokenizer for training and inference.
✅ **Define Pad Token**: Some models (like GPT-2/Llama) may not have a default pad token.
   ```python
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```
✅ **Attention Masks**: Always pass the attention mask to the model to avoid learning noise.
✅ **Check Direction**: Use right padding for training and left padding for generation.
