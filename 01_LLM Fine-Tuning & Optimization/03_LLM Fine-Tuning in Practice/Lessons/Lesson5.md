# Fine-Tuning LLMs on RunPod: Your Cloud GPU Training Setup

You've fine-tuned in Colab — now it's time to upgrade your infrastructure.

Colab is great for learning, but it has limits: session timeouts, GPU availability issues, and environments that reset every time you reconnect. **RunPod** gives you more control: dedicated GPUs, persistent storage, and the freedom to run training jobs for hours or days without interruption.

In this lesson, you'll set up your first RunPod environment and run your fine-tuning code end-to-end. You'll learn SSH access, workspace management, environment configuration, and how to execute training scripts.

## From Colab to Cloud: Why Move to RunPod?
You don’t need to change your workflow — just your infrastructure. The same scripts and datasets you used in Colab will work on RunPod, only faster and more reliably.

Think of it as moving from a temporary bench in a shared lab to your own workstation.

## Meet RunPod: Your Dedicated GPU Workspace
RunPod offers a full AI lifecycle suite:
* **Pods**: On-demand cloud GPUs (what we’ll use).
* **Clusters**: Multi-node GPU clusters for large-scale training.
* **Serverless**: Auto-scaling GPU endpoints for inference.
* **Hub**: Instantly deploy open-source AI projects.

### Why Pods?
* **Flexible**: Choose your GPU (A40, A100, H100).
* **Persistent**: Data in `/workspace` stays put between stops.
* **Dev-Friendly**: Full root access via SSH.

🎥 **Video Walkthrough: Getting Started with RunPod**
Deploying your first GPU pod from scratch.

---

## Setting Up Your Environment

### 1. SSH Access (Crucial for Stability)
SSH gives you a stable, secure connection — ideal for long runs.

**Generate a key (Windows PowerShell or Mac/Linux terminal):**
```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```
**Copy your public key:**
* Windows: `cat ~/.ssh/id_rsa.pub | clip`
* Mac/Linux: `cat ~/.ssh/id_rsa.pub`

**Add it to RunPod**: Go to **Settings → SSH Keys** and paste your public key.

---

## Launching and Connecting

1. **Deploy**: Pods → Deploy New Pod.
2. **Select GPU**: **NVIDIA A40** is a great balance of performance ($0.40/hr) and memory (48GB).
3. **Template**: Use the default **PyTorch** + CUDA image.
4. **Connect**: Copy the SSH command provided by RunPod.

```bash
ssh root@ssh.runpod.io -p <your-port> -i ~/.ssh/id_rsa
```

### Working Inside the Pod
Always move to the persistent directory:
```bash
cd /workspace
git clone https://github.com/your-repo/llm-engineering
cd llm-engineering
pip install -r requirements.txt
```

🎥 **Video Walkthrough: Running Your Code on RunPod**
Executing training scripts and managing workspace sessions.

---

## Pro Tips for Cloud Training

### Persistence
Only files in `/workspace` are saved when you stop the pod. Everything else (like things in `/home` or `/tmp`) is wiped!

### Keeping Sessions Alive (tmux)
Use `tmux` so your training continues even if you close your terminal or lose internet:
```bash
# Start session
tmux new -s training

# Run your script
python train.py

# Detach: Ctrl+B, then press D
# Reattach: tmux attach -t training
```

### Managing Costs
* **Stopping a Pod**: Stops billing for the GPU, but you still pay a small storage fee (~$0.10/hr) for your `/workspace`.
* **Terminating a Pod**: Stops all billing and **deletes all data**.

**Environment Check**:
```python
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

Next, we'll use this infrastructure to run **systematic experiments** with experiment tracking.
