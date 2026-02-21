
### The Three Key Decisions for Fine-Tuning

Before starting, every engineer must make three key choices that define their workflow, control, cost, and scalability.

---

### Decision Layer 1: Model Access (Frontier vs. Open-Weight)

This is about *which kind* of model you will train.

**1. Frontier Models (API-Based)**
* **What They Are:** Closed-weight models from large providers (e.s., GPT-4, Claude, Gemini).
* **How it Works:** You fine-tune via the provider's API. You upload a dataset (e.g., JSONL) and they handle the training "behind the scenes." You never touch the model weights.
* **Pros:** Simple, scalable, and requires no infrastructure management.
* **Cons:**
    * No control over the architecture or training process.
    * Not reproducible or auditable.
    * You pay per request (often at a higher rate).
* *Covered in Week 3 (OpenAI, Gemini Studio, etc.).*

**2. Open-Weight Models (Full Control)**
* **What They Are:** Downloadable and customizable models (e.g., LLaMA 3, Mistral, Phi-3).
* **How it Works:** You download the weights and train them on your own hardware (local or cloud).
* **Pros:**
    * Full control and transparency.
    * Enables deep customization and auditing.
    * Independent of any single provider.
* **Cons:** You are responsible for managing compute, experiments, and reproducibility.
* *This is the **primary focus** of Modules 1 and 2 in the program.*

---

### Decision Layer 2: Compute Environment (Local vs. Cloud)

This is about *where* your training code will run.

**1. Local Training**
* **What it is:** Running training scripts on your own workstation or internal servers (or a platform like Google Colab).
* **Pros:**
    * Full control and data privacy.
    * Ideal for small experiments and rapid iteration.
* **Cons:** Limited by your local GPU capacity.
* *Covered in Week 2 (using Google Colab).*

**2. Cloud Training**
* **What it is:** Renting scalable GPU resources on demand from providers like AWS EC2, RunPod, Vast.ai, etc.
* **How it Works:** You run the *same training scripts* (e.g., from Hugging Face) but on more powerful, remote hardware.
* **Pros:** Scalable to handle large models and datasets.
* **Cons:** Incurs compute costs.
* *Covered in Module 2 (using services like AWS SageMaker).*

---

### Decision Layer 3: Orchestration (Custom Code vs. Managed Framework)

This is about *how* you will manage the training process.

**1. Custom Code Approach**
* **What it is:** Working directly with foundational libraries like **Hugging Face `Transformers`**, `PEFT` (Parameter-Efficient Fine-Tuning), `TRL` (Transformer Reinforcement Learning), and `Accelerate`.
* **How it Works:** You write Python scripts to control every detail: LoRA parameters, learning rates, checkpointing, etc.
* **Pros:**
    * Maximum control and flexibility.
    * Ideal for research and experimenting with new techniques (like QLoRA).
* **Cons:** More complex and requires deeper technical knowledge.
* *Covered in Week 3.*

**2. Managed Framework Approach**
* **What it is:** Using a higher-level framework that abstracts away the complex code. You define your training in a configuration file (e.g., YAML).
* **Examples:** **Axolotl**, AWS SageMaker, Together.ai.
* **How it Works:** You specify the base model, dataset, and parameters in a config file, and the framework handles the orchestration, scaling, and logging.
* **Pros:**
    * Simpler and more automated.
    * Ideal for production workflows needing reliability and consistency.
* **Cons:** Less flexible than writing custom code.
* *Covered in Week 3 (Axolotl) and Week 5 (Bedrock).*

---

### Summary Table: Choosing Your Workflow

| If you value... | Frontier (API) | Open-Weight | Local | Cloud | Custom Code | Managed Framework |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simplicity & Speed** | ✅ | | | ✅ | | ✅ |
| **Transparency & Control** | | ✅ | ✅ | | ✅ | |
| **Cost Efficiency** | | ✅ | ✅ | | | ✅ |
| **Scale & Performance** | ✅ | | | ✅ | | ✅ |
| **Flexibility & Experimentation**| | ✅ | ✅ | | ✅ | |
| **Reliability & Automation** | ✅ | | | ✅ | | ✅ |
| **Reproducibility & Auditing** | | ✅ | ✅ | | ✅ | ✅ |

**Note:** Most real-world projects use a **hybrid approach**: experimenting locally (Custom Code), scaling on the cloud (Managed Framework), and deploying.