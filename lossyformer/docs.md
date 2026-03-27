# What does it do?

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


## Supported Models

| Model | Architecture | Early Exit Wrapper | Tested |
|-------|-------------|-------------------|--------|
| `bert-tiny` | BERT | `BertEarlyExit` | IMDB (77% baseline) |
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