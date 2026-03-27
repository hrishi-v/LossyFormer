import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification

from gelu_to_relu import (
    _to_masegraph,
    _swap_gelu_for_relu,
    train_and_save_model,
    get_tokenized_mnli,
    make_eval_fn,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = [
    ("FacebookAI/roberta-base", True),
    ("FacebookAI/roberta-large", True),
]


def train_pipeline(checkpoint, dataset, tokenizer, needs_baseline_training=False):
    model_name = checkpoint.split("/")[-1]
    home = Path.home()

    print(f"\n===== {model_name} =====")

    eval_fn = make_eval_fn(dataset, tokenizer, DEVICE, eval_split="validation_matched")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3
    ).to(DEVICE)
    model.config.problem_type = "single_label_classification"

    hf_dir = home / f"{model_name}-model-hf"
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    print(f"Saved HF baseline → {hf_dir}")

    train_and_save_model(
        model,
        dataset,
        tokenizer,
        f"{home}/{model_name}-model",
        do_train=needs_baseline_training,
    )

    print("Baseline accuracy:", eval_fn(model))

    mg = _to_masegraph(model)
    mg, _ = _swap_gelu_for_relu(mg)
    relu_model = mg.model

    print("After GELU→ReLU accuracy:", eval_fn(relu_model))

    mg.export(f"{home}/{model_name}-model-no-gelu")

    train_and_save_model(
        relu_model,
        dataset,
        tokenizer,
        f"{home}/{model_name}-model-no-gelu-trained",
        do_train=True,
        apply_relu_swap=True,
    )


def main():
    for checkpoint, needs_training in CHECKPOINTS:
        dataset, tokenizer = get_tokenized_mnli(
            checkpoint, return_tokenizer=True
        )
        train_pipeline(
            checkpoint,
            dataset,
            tokenizer,
            needs_baseline_training=needs_training,
        )


if __name__ == "__main__":
    main()