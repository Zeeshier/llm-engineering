# LoRA Hyperparameter Tuning: Analyzing Systematic Fine-Tuning Experiments

You’ve built a complete fine-tuning pipeline. You’ve trained your first LoRA model. You’ve set up cloud infrastructure and experiment tracking. Now comes the next challenge: **which hyperparameters actually work best?**

In this lesson, we’ll look at results from a structured set of fine-tuning experiments — run on RunPod, tracked in Weights & Biases, and analyzed across multiple configurations.

## The Experiments We Ran
We focused on the three core questions that almost always shape fine-tuning performance:

1.  **Capacity**: How much "learning bandwidth" do the adapters need? ($r = 4, 8, 16, 32$)
2.  **Speed**: How fast should the model learn? (Learning rate $2e-5$ vs. $2e-4$)
3.  **Targeting**: Which parts of the model should we fine-tune? (q+v, full attention, or attention+MLP)

Every run used the full SAMSum dataset (~14K samples) and evaluation on a shared 200-sample validation split.

🎥 **Video Walkthrough: LoRA Fine-Tuning Experiments and W&B Tracking**
Analyzing multiple configurations to find the optimal setup.

---

## Experiment 1: LoRA Rank (Capacity)
LoRA rank ($r$) determines how expressive the adapters are.

| LoRA Rank | ROUGE-1 | ROUGE-2 | ROUGE-L | Δ vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| $r = 4$ | 45.70% | 22.04% | 38.16% | -0.72% |
| **$r = 8$ (Base)** | **46.74%** | **23.17%** | **38.88%** | **baseline** |
| **$r = 16$** | **47.27%** | **23.29%** | **39.25%** | **+0.37%** ✅ |
| $r = 32$ | 47.33% | 23.10% | 38.99% | +0.11% |

![LoRA Rank Analysis](hpt_lora_rank.jpeg)

> [!TIP]
> **Recommendation**: Use **$r = 16$** for the best balance of performance and efficiency. Beyond this, benefits begin to plateau.

---

## Experiment 2: Learning Rate (Speed)
The difference here was the most dramatic.

| Learning Rate | ROUGE-1 | ROUGE-2 | ROUGE-L | Δ vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| $2e-5$ | 43.27% | 19.19% | 34.83% | -4.05% ❌ |
| **$2e-4$** | **46.74%** | **23.17%** | **38.88%** | **baseline** |

![Learning Rate Analysis](hpt_learning_rate.jpeg)

> [!WARNING]
> At $2e-5$, the model barely moves. $2e-4$ provides smooth convergence without instability for single-epoch runs.

---

## Experiment 3: Target Modules (Targeting)
Where should we put the adapters?

| Target Modules | ROUGE-1 | ROUGE-2 | ROUGE-L | Δ vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| q + v | 46.74% | 23.17% | 38.88% | baseline |
| **Full attention** | **47.41%** | **24.06%** | **39.55%** | **+0.67%** ✅🏆 |
| Attention + MLP | 47.62% | 23.73% | 39.23% | +0.35% |

![Target Modules Analysis](hpt_target_modules.jpeg)

---

## The Winning Configuration
The optimal setup for Llama 3.2 1B on SAMSum:

*   **LoRA Rank**: 16
*   **LoRA Alpha**: 32
*   **Learning Rate**: $2e-4$
*   **Target Modules**: q, k, v, o (Full Attention)
*   **ROUGE-L**: **39.55%**

## Putting It All in Perspective
| Model | Type | ROUGE-L |
| :--- | :--- | :--- |
| Llama 3.2 1B (Base) | Open Baseline | 27.24% |
| GPT-4o-mini (Base) | Frontier Baseline | 32.91% |
| **Llama 3.2 1B (Optimized)** | **Self-hosted FT** | **39.55%** |
| GPT-4o-mini (Fine-tuned) | Managed FT | 45.96% |

Your optimized 1B model now outperforms the **base** GPT-4o-mini — a much larger frontier model — while remaining fully private and reproducible.

### Summary of Week 3
You’ve moved from **intuition to evidence**. You didn't just fine-tune; you built a repeatable, production-grade system. You now have the skills to optimize any model, on any task, with data-driven confidence.
