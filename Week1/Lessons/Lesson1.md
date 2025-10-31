# 🏛️ The Foundation: The Transformer

* All modern LLMs (like GPT, Claude, and Gemini) are built on the Transformer architecture, introduced in the 2017 paper "Attention Is All You Need."

* Its key innovation was attention, allowing the model to analyze all words in a sentence at once to understand context (e.g., distinguishing "river bank" from "money bank").

* The original Transformer had two parts: an Encoder (to read and understand input) and a Decoder (to generate output).

## 🧬 The Three Main Architectures
Researchers adapted the original Transformer into three "families" for different tasks.

### 1. Encoder-Decoder (The Translator)

**What it is**: The original, full Transformer model.

**How it works**: The Encoder reads an input sequence, and the Decoder transforms it into a new output sequence.

**Best for**: Sequence-to-sequence tasks like translation and summarization.

**Example Models**: T5, BART, PEGASUS.

### 2. Encoder-Only (The Analyst)

**What it is**: Uses only the Encoder part.

**How it works**: Reads the entire input text at once (bidirectionally) to build a deep understanding.

**Best for**: Understanding and classification tasks like sentiment analysis, topic detection, or Named Entity Recognition (NER).

**Key Point**: It cannot be used to generate fluent, new text (like a chatbot).

**Example Models**: BERT, RoBERTa, DistilBERT.

### 3. Decoder-Only (The Author)

**What it is**: Uses only the Decoder part. This is the dominant architecture for modern generative AI.

**How it works**: Generates text one token (word) at a time, from left to right. It is autoregressive, meaning each new word is based on all the words that came before it.

**Best for**: Generation and conversation, including chat assistants, code generation, and creative writing.

**Example Models**: GPT-3.5/4, Claude 3, Gemini, LLaMA, Mistral.

### ⭐ Why Decoder-Only Models Dominate
This program focuses on Decoder-Only models because they have become the industry standard.

**Versatility**: A single model can perform tasks from all three categories (chat, translation, summarization, and even classification).

**Scalability**: They scale predictably—more data and a larger model lead to better performance.

**Ecosystem Maturity**: The vast majority of open-source tools (like Hugging Face, PEFT, and Axolotl) are built to support decoder-only models.

**Key takeaway**: When people refer to an "LLM" today for a task like building a chatbot or assistant, they are almost always talking about a Decoder-Only model.