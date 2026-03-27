# LossyFormer

**Automated lossy compression for transformer models through joint head pruning, LoRA recovery, and early exit.**

LossyFormer takes a fine-tuned HuggingFace transformer and produces a smaller, faster version that stays within a user-specified accuracy budget. It combines three techniques in an iterative loop: structured attention head pruning guided by Taylor importance scores, LoRA-based fine-tuning for accuracy recovery, and entropy-based early exit to skip unnecessary computation at inference time.

## How It Works

LossyFormer operates as an iterative optimize-and-recover loop:

1. **Baseline evaluation** — measure the uncompressed model's accuracy and throughput.
2. **Early exit classifier training** — train lightweight classifiers at each layer to estimate when a sample can exit early (based on prediction entropy).
3. **Iterative pruning loop** — repeat until the accuracy budget is exhausted:
   - **Survival calibration** — estimate which layers are actually reached at inference time (via multi-threshold entropy analysis), producing per-layer execution probabilities.
   - **Head importance profiling** — attach forward hooks to compute Taylor expansion importance scores (`|gradient × activation|`) for each attention head, weighted by the layer's survival probability. Heads in layers that are rarely reached get deprioritized.
   - **Structured pruning** — remove the lowest-scoring 10% of heads using HuggingFace's native `prune_heads()`, which cleanly resizes Q/K/V/output projection weights.
   - **LoRA recovery** — fine-tune the pruned model with low-rank adapters (rank 32) to recover accuracy lost from pruning, then merge adapters back into the base weights.
   - **Threshold sweep** — evaluate multiple early exit thresholds and select the one that maximizes throughput while staying above the target accuracy.
4. **Return the optimized model** — an early exit wrapper around the pruned backbone, ready for inference.

The key insight is that pruning and early exit interact: removing heads from deep layers that are rarely reached (due to early exit) costs almost nothing, while pruning heads in early layers that every sample passes through is expensive. The survival-weighted importance scoring captures this interaction.

## Project Structure

```
lossyformer/
├── main.py                  # LossyFormer class — the main optimization loop
├── utils.py                 # eval_accuracy, eval_speed utilities
├── early_exit/
│   ├── __init__.py          # get_early_exit_model dispatcher
│   ├── EarlyExitBase.py     # Shared mixin: confidence evaluation, logit routing
│   ├── BertEarlyExit.py     # BERT-specific early exit wrapper
│   └── RobertaEarlyExit.py  # RoBERTa-specific early exit wrapper
├── pruning/
│   ├── __init__.py
│   ├── HeadProfiler.py      # HeadProfiler class for importance score collection
│   ├── pruning.py           # Hooks, survival calibration, pruning decisions
│   └── finetune.py          # LoRA fine-tuning and adapter merging
└── tests/
    ├── test_head_pruning.py  # Unit tests for profiling, pruning, multi-iteration cycles
    └── test_module.py        # Unit tests for early exit wrappers, LossyFormer integration

lf_tests/
├── lf-testing.py            # Sweep script across accuracy drop targets
└── quickstart.py            # Minimal usage example
```

## Quickstart

```python
import torch
from transformers import AutoModelForSequenceClassification
from lossyformer.main import LossyFormer

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Load your fine-tuned weights
state_dict = torch.load("path/to/your/checkpoint.pt", map_location="cuda")
model.load_state_dict(state_dict, strict=False)

lossy = LossyFormer(allowed_accuracy_loss=0.02, device="cuda")

optimized_model = lossy.fit(
    model_name,
    model,
    max_ft_steps=500,
    dataset_name="glue",
    dataset_config="mnli",
    tokenizer_name=model_name,
    text_columns=["premise", "hypothesis"],
)

if optimized_model is not None:
    print(f"Threshold: {optimized_model.threshold}")
    print(f"Pruning iterations: {len(lossy.iteration_history)}")
```

You can also pass your own data loaders directly instead of letting LossyFormer build them:

```python
optimized_model = lossy.fit(
    model_name,
    model,
    train_loader=your_train_loader,
    eval_loader=your_eval_loader,
    max_ft_steps=500,
)
```

See `lf_tests/quickstart.py` for a complete runnable example with swappable configurations for BERT/MNLI, RoBERTa/MNLI, and IMDB.

## Running the Sweep

The sweep script evaluates LossyFormer across multiple accuracy drop targets (0.5%, 1%, 2%, 4%, 7%, 10%) and logs results to CSV:

```bash
PYTHONPATH=. python lf_tests/lf-testing.py --model bert-base --max_ft_steps 500
PYTHONPATH=. python lf_tests/lf-testing.py --model roberta --max_ft_steps 500
```

Output CSV columns: target drop %, final accuracy, latency, parameter count, early exit threshold, total head reduction %.

## Running Tests

```bash
# Unit tests for pruning logic
python -m pytest lossyformer/tests/test_head_pruning.py -v

# Unit tests for early exit and LossyFormer integration
python -m pytest lossyformer/tests/test_module.py -v

# All tests
python -m pytest lossyformer/tests/ -v
```

Tests use tiny 2-layer models (4 heads, hidden size 64) that run in seconds on CPU.

## Supported Models

| Model | Architecture | Early Exit Wrapper | Tested |
|-------|-------------|-------------------|--------|
| `bert-base-uncased` | BERT | `BertEarlyExit` | MNLI (82% baseline) |
| `roberta-base` | RoBERTa | `RobertaEarlyExit` | MNLI (87% baseline) |

Adding a new architecture requires implementing an early exit wrapper (inheriting from `EarlyExitBase`) and registering it in `early_exit/__init__.py`. The pruning infrastructure works with any HuggingFace model that has `encoder.layer[i].attention.self` structure.clear

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `allowed_accuracy_loss` | 0.01 | Maximum accuracy drop from baseline (e.g., 0.02 = 2%) |
| `max_ft_steps` | 500 | LoRA fine-tuning steps per pruning iteration |
| `step_keep_ratio` | 0.90 | Fraction of heads to keep per iteration (prune 10%) |
| `max_iterations` | 25 | Maximum pruning iterations before stopping |
| `init_epochs` | 100 | Batches for initial early exit classifier training |

## Dependencies

Requires Python ≥ 3.11 and CUDA for GPU acceleration. Core dependencies: `torch`, `transformers`, `peft`, `datasets`, `tqdm`.

```bash
uv sync
```