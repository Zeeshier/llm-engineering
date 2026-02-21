# LLM Fine-Tuning Foundations: Understanding Next-Token Prediction

Last week, you explored the landscape of large language models — architectures, ecosystems, and when to fine-tune versus use RAG.
Now we’re shifting gears to focus on something more fundamental: how fine-tuning actually works.

To understand that, we first need to zoom in on how language models learn in the first place.
At the core of that learning process is a single, powerful idea: next-token prediction.

Every training step, every fine-tuning run, every loss curve you’ll see later — it all traces back to this one mechanism.

That’s the key insight here: language models aren’t mysterious text generators — they’re enormous classifiers trained to predict what comes next, choosing from tens of thousands of possible tokens at every step.

And once you see them that way, everything about fine-tuning starts to make sense.

## The Big Idea: Predicting What Comes Next
Let’s start with something familiar.
If you’ve ever used predictive text on your phone, you’ve already seen a miniature version of what language models do.

Type “The cat sat on the…” and your phone might suggest “mat.”
Simple enough. But behind that tiny prediction lies the same principle that powers billion-parameter models.

Here’s what’s really happening: when the model predicts “mat,” it isn’t choosing from just a few options.
It actually computes probabilities for every single token in its vocabulary — words, subwords, symbols, and even awkward or irrelevant ones like “mitochondrion,” “chemistry,” or “with.”

Considering that a typical language model’s vocabulary can include around 50,000 tokens, each prediction involves scoring all of them before picking the next one.

Let’s visualize that:

| Candidate Token | Probability |
| :--- | :--- |
| mat | 0.78 |
| floor | 0.12 |
| chair | 0.05 |
| bed | 0.02 |
| table | 0.01 |

The model doesn’t know the answer. It computes probabilities for every possible token — even nonsensical ones — and then chooses one, either by selecting the highest probability or sampling from the distribution.

That single classification decision — which token should come next — is the core of everything that follows in fine-tuning and beyond.

### Video Walkthrough: How Language Models Predict the Next Token
In this video, you’ll see how large language models convert text generation into a massive classification problem — predicting the next token from tens of thousands of possibilities at every step.

## From Classification to Generation: The Autoregressive Loop
So far, we’ve focused on how a model makes a single prediction.
Now let’s connect that to how it generates entire paragraphs or conversations.

Language models are autoregressive, meaning they use their own previous outputs as inputs for the next prediction.
Think of it as a self-feeding loop.

Let’s walk through an example step by step:

**User**: What’s the capital of France?
**Assistant**:
1. The model predicts “The” as the next token.
2. Then it feeds “The” back in and predicts “capital.”
3. Now it sees “The capital” and predicts “of.”
4. Then “France.”
5. Then “is.”
6. Then “Paris.”

Each step builds on everything before it — just like you might build a sentence in your head word by word.
But remember: the model isn’t planning ahead. It’s only ever predicting one token at a time, given the full context so far.

That’s why this process is so powerful and so fragile.
Every new token depends entirely on the history it just created.
One wrong prediction can change the entire trajectory of the response.

🎥 **How Autoregression Works in Large Language Models**
In this video, you’ll see how large language models generate text by feeding their own predictions back as input — turning simple next‑token prediction into full sentence generation.

## Why Model Outputs Vary: Probabilistic Sampling
If you’ve ever asked ChatGPT the same question twice and got slightly different answers, you’ve seen probabilistic sampling in action.

When generating text, the model doesn’t always pick the most likely token.
Instead, it samples from the distribution to introduce controlled randomness — the same way a human writer might choose “quick,” “rapid,” or “swift” depending on tone.

This is controlled through parameters like:

* **Temperature** — how “bold” the sampling is. Low temperature makes the model predictable; high temperature makes it creative.
* **Top-k and Top-p (nucleus) sampling** — methods for choosing from the top portion of likely tokens while ignoring unlikely ones.

This combination explains why the same prompt can yield new phrasing each time.
It’s not inconsistency — it’s intentional variety built into the decoding process.

Later in the program, we’ll revisit these settings when we explore inference and generation control.

## Pattern Matching, Not Reasoning
Here’s the trap many people fall into:
because language models sound like they’re reasoning, it’s easy to forget they’re not.

They don’t “think.” They match patterns.

When a model writes an explanation, it isn’t reasoning through logic.
It’s predicting what text usually follows similar text it has seen before.

For example, if it’s seen millions of examples of people solving math problems step by step, it’s learned that pattern. So when prompted, it generates “reasoning-like” sequences — but without real understanding.

This is crucial for fine-tuning.
You’re not teaching the model new reasoning abilities — you’re shaping which patterns it reproduces.

## How Fine-Tuning Builds on This
Now that we’ve seen how models predict and generate, let’s connect this back to fine-tuning.

Fine-tuning doesn’t alter the learning algorithm.
It doesn’t teach the model how to learn — it changes what it learns from.

Here’s what shifts:

* **The data format**: from raw text to structured examples like instructions or dialogues.
* **The masking strategy**: deciding which parts of the text actually contribute to loss (we’ll cover this soon).
* **The scale**: from trillions of pretraining tokens to thousands or millions of curated fine-tuning examples.

The mechanism — predicting the next token — stays exactly the same.
You’re simply giving the model new kinds of patterns to learn from.

Think of it like a chef who already knows how to cook — fine-tuning doesn’t teach them to hold a knife again; it teaches them a new cuisine.

Understanding this foundation sets up everything that comes next.

You’ll often hear the terms training, pretraining, and fine-tuning used as if they describe different processes — but technically, they’re the same.

In all cases, the model updates its weights by comparing predicted tokens to the correct ones and minimizing the loss.
The only real difference lies in what data and what purpose the training serves:

* **Pretraining**: The model learns general language patterns from a massive, diverse corpus.
* **Fine-tuning**: We continue the same process, but on a smaller, specialized dataset — for example, medical notes, legal documents, or chat-style conversations.

In other words, “fine-tuning” isn’t a new algorithm — it’s just training again, with a different goal and a more focused dataset.
We use the term “fine-tuning” because it captures how we use the process: to refine a broad model for a narrower purpose.
