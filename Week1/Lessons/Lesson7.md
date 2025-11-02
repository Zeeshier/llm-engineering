Reproducing Benchmarks 
---------------------------

This exercise shows you how to locally run a benchmark from the **Hugging Face Open LLM Leaderboard**. We'll use the same tool Hugging Face uses: **lm-evaluation-harness**. This makes your evaluation transparent, consistent, and reproducible.

We will test the meta-llama/Llama-3.2-1B-Instruct model on the **tinyGSM8K** benchmark, which measures math reasoning.

### Step 1: Set Up Colab Environment

1.  Open a new Google Colab notebook.
    
2.  Change the runtime to **GPU** (Runtime -> Change runtime type -> GPU).
    
3.  Bash!pip install lm\_eval langdetect -q!pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
    
4.  Bash!lm\_eval --help
    

### Step 2: Run Evaluation (Command-Line)

This is the quickest way to test a model.

```Bash
!lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --tasks tinyGSM8K \
    --device auto \
    --batch_size auto


```

This command will:

*   Download the specified model from Hugging Face.
    
*   Load it onto the GPU.
    
*   Run the tinyGSM8K benchmark.
    
*   Print the final accuracy, which you can compare to the official leaderboard.
    

### Step 3: Run Evaluation (Python API)

Using the Python API is better for integrating evaluation into your code, like after a fine-tuning job.

```Python

from lm_eval import evaluator
from joblib import dump


results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.2-1B-Instruct,parallelize=True,trust_remote_code=True",
    tasks=["tinyGSM8K"],
    device="cuda",
    batch_size="auto"
)


# Print and save the results
print(results)
dump(results, "results.joblib")

```


This saves the structured results, allowing you to track performance improvements over time.

### Step 4: Interpret Results

The output will give you an **accuracy score**. For tinyGSM8K, the meta-llama/Llama-3.2-1B-Instruct model scores around **39%**.

*   **strict vs. flexible metrics:** strict counts only _exact_ matches, while flexible allows for minor formatting differences.
    
*   **Manual Inspection:** The tool also allows you to see the model's generated answer versus the true answer, helping you understand _why_ it failed or succeeded.
    

You will use this exact process later to verify if your own fine-tuned models are actually getting better.