# How LLMs Learn: Loss, Masking, and Next-Token Prediction

You've seen that language models are massive classifiers predicting the next token.

But here's the question we haven't answered yet: **How do they get good at it?**

This lesson shows you the mechanism behind all LLM learning — whether it's pre-training GPT on the internet or fine-tuning Llama on your company's data.

You'll learn three core concepts that work together: **next-token prediction** (the task), **cross-entropy loss** (the measurement), and **masking** (what to learn from). They form the foundation for every fine-tuning technique you'll use in this program.

By the end, you'll understand exactly what happens when a model "learns" from text, and why different masking strategies produce different model behaviors — from base models that complete sentences to chat models that follow instructions.

## From Prediction to Learning
In the last lesson, we treated large language models as classifiers — predicting the next token out of tens of thousands of possibilities.
That’s what they do during inference: take context, predict the next token, repeat.

But there’s a deeper question hiding under that surface:
**how did they learn to predict so well in the first place?**

The answer lies in a simple feedback loop.
Every model starts out terrible — guessing tokens almost at random. During training, it compares its guesses to the correct tokens, measures how wrong it was, and adjusts its parameters to make better predictions next time.

That process — prediction, comparison, and adjustment — is what turns a random model into a capable one.
Let’s unpack how that actually happens.

## Choosing The Training Objective
We already know the model’s goal: predict the next token given everything it has seen so far.
That single objective is surprisingly powerful. It’s how a model learns grammar, style, reasoning, even facts about the world — just by predicting what typically comes next.

Imagine training on this simple sequence:
*“The cat sat on the …”*

During training, the model doesn’t just see the sentence.
It sees a tokenized version of it — numbers representing words or subwords — and tries to predict each token in turn.

| Input Sequence | Target (Next Token) |
| :--- | :--- |
| The | cat |
| The cat | sat |
| The cat sat | on |
| The cat sat on | the |
| The cat sat on the | mat |

At each step, the model produces a probability distribution over its entire vocabulary — maybe 50,000 possible tokens.
If “mat” gets 0.78 probability and every other token gets less, that’s a good prediction.
If it gives 0.01 to “mat” and 0.70 to “chair,” that’s a mistake.

The next question is: **how does it know how big a mistake that was?**

## Measuring Wrong: Cross-Entropy Loss
Learning starts with being wrong — but you need a way to measure wrongness.
That’s what the loss function does. It takes the model’s predicted probabilities and the correct next token, and computes how far apart they are.

In language modeling, that function is almost always **cross-entropy loss**.

Formally, for a single token, the loss is calculated as:
$$L = -\sum_{c=1}^{V} y_c \log(p_c)$$

where:
* $V$ is the vocabulary size
* $y_c$ is the true label (1 for the correct token, 0 for all others)
* $p_c$ is the model’s predicted probability for token $c$

Since only the correct token has $y_c = 1$, this simplifies to:
$$L = -\log(p_{\text{target}})$$

where $p_{\text{target}}$ is the model’s predicted probability assigned to the true next token.

### Example
Let’s say the model predicts the next token probabilities after *“The cat sat on the …”* like this:

| Candidate Token | Predicted Probability |
| :--- | :--- |
| mat | 0.78 |
| floor | 0.12 |
| chair | 0.05 |
| bed | 0.03 |
| wall | 0.02 |

The correct answer is “mat.”
Cross-entropy loss looks at the probability assigned to that token (0.78) and converts it into a penalty using the negative log:
$$-\log(0.78) \approx 0.25$$

Lower loss means better prediction.
If the model had predicted 0.99 for “mat,” the loss would be much smaller (0.01); if it had predicted 0.01, the loss would be much higher (4.61).

This loss value for the predicted token becomes the model’s training signal.
It’s how the model knows whether to strengthen or weaken the internal connections (weights) that led to that prediction.

## From One Token to a Sequence
A real training example is longer than one token, so the model repeats this prediction process for every position in the sequence.

For example, if your sentence is:
*“The cat sat on the mat.”*

the training process looks like this:

| Input Sequence | Target (Next Token) |
| :--- | :--- |
| The | cat |
| The cat | sat |
| The cat sat | on |
| The cat sat on | the |
| The cat sat on the | mat |

At each step, the model predicts the next token based on the true preceding tokens given to it, compares its prediction to the actual token, and calculates a loss.

The model **does not** append its own predicted token to the sequence during training — that autoregressive behavior happens only during inference.

This approach, known as **teacher forcing**, ensures the model learns from the correct context rather than compounding its own mistakes.

The per-token losses are then averaged to produce a sequence-level loss, which is used to update the model’s weights.

After millions of such examples, small improvements accumulate — and the model gets steadily better at predicting the next token across diverse contexts.

## When the "Right" Answer Gets Penalized
Here's an interesting nuance of token-level loss.

Suppose the training data contains:
**Question**: What is the capital of France?
**Answer**: The capital of France is Paris.

During training, the model sees *"What is the capital of France?"* and must predict the next token.

Let's say it assigns:
* 0.5 probability to "Paris"
* 0.3 probability to "The" (the actual next token in the training example)

The loss is calculated on "The":
$$-\log(0.3) \approx 1.20$$

Wait — the model gave higher confidence to "Paris," which is semantically correct! Why is it being penalized?

Because the model isn't judged on meaning — it's judged on matching the exact training sequence.

In this case, the training example happened to start with "The capital of…" rather than "Paris…". Both are valid responses, but only one matches the training data.

This might seem like a flaw, but it's actually fine. Here's why:
The model sees billions of examples. Some will format answers one way, others differently. Across that massive distribution, the model learns the deeper patterns — that Paris is associated with France's capital — even if individual examples penalize "correct but differently formatted" predictions.

**Loss is local, learning is global.** Any single token prediction might feel arbitrary, but averaged over billions of training steps, these signals converge on coherent language understanding.

This is a key insight: language models learn from statistical patterns across vast data, not from individual examples being "fair."

## Masking: Deciding Which Tokens Matter
So far, we’ve treated every token as equally important.
In practice, not every token should contribute to the loss.

That’s where **masking** comes in.

Masking has two purposes:
1. It controls what each token can see when making predictions (**causal masking**).
2. It controls which tokens actually count toward the loss (**assistant-only masking** and other selective strategies).

Together, these determine both the model’s behavior and what it learns to optimize.

### 1. Causal Masking (Base Models)
Causal masking ensures the model can’t see future tokens while predicting.
That’s what keeps it autoregressive — predicting left to right.

If your training sequence is:
*“The cat sat on the mat.”*

The model can use “The cat sat” to predict “on,” but not “on the” to predict “sat.”
Visually, you can imagine a triangular mask — each token can only attend to itself and earlier tokens.

This maintains the same behavior the model will use at inference time: generate one token, append it, and move forward.

### 2. Assistant-Only Masking (Chat Models)
When training instruction-following or chat models, we still apply causal masking — the model continues to predict tokens left to right.

However, we only compute loss on the assistant’s tokens, not the user’s.

For example:
**User**: What's the capital of France?
**Assistant**: The capital of France is Paris.

Here, causal masking ensures the model predicts each assistant token using only the ones before it, but assistant-only masking ensures the loss is applied only on the assistant’s part:
*“The capital of France is Paris.”*

The model doesn’t learn to predict the user’s input — it learns to generate the appropriate reply, step by step.

So, **assistant-only masking = causal masking + selective loss scoring**.

This distinction — what gets scored — is what separates base models that complete text from instruction-tuned models that follow prompts.

We’ll revisit masking (and its partner, label shifting) in detail in a dedicated lesson later this week.

## The Power of Selective Scoring
This idea of selective scoring is incredibly powerful.
By choosing which parts of a sequence to apply loss on, you can shape what kind of model you train.

That’s how we turn a base model into a chat model — simply by masking out user inputs and focusing the loss on assistant responses.

And this concept extends far beyond chat.
For example, you could design a model that reasons step-by-step before giving an answer, and apply the loss only on the reasoning tokens to teach better logical decomposition.
Or you could score both reasoning and final answers differently to balance interpretability and precision.

Masking isn’t just about excluding tokens — it’s a fine-grained control over what kind of behavior the model learns to optimize.

## The Learning Loop: How Everything Fits Together
Let’s put it all together.

Training a language model follows a repeating loop:
1. Take a batch of text.
2. Predict the next token at every position.
3. Compute the loss (using cross-entropy).
4. Apply masking so only valid tokens count.
5. Update model weights to reduce the loss.
6. Then repeat — billions of times.

That’s it.
This cycle — **prediction → loss → masking → update** — is the engine behind every base model, every fine-tuning run, and every alignment method built on top of them.

The same process powers both training and fine-tuning.
What changes are the datasets used and the masking or selective scoring strategies that determine what the model learns to focus on.

* **In pretraining**, the model learns general language patterns from massive, diverse text corpora.
* **In fine-tuning**, it applies that same learning process to more targeted datasets — whether that’s instruction–response pairs, medical notes, legal documents, or scientific text — adapting its behavior for the specific goals you define.

That’s how a single training mechanism gives rise to everything from general-purpose assistants to highly specialized domain models.
