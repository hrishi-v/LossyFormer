import torch
from pathlib import Path

from gelu_to_relu import (
    run_inference_benchmark,
    save_and_print_results,
    make_eval_fn,
    get_tokenized_mnli,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = [
    "FacebookAI/roberta-base",
    # "FacebookAI/roberta-large",
]

def main():
    all_results = []

    for checkpoint in CHECKPOINTS:
        model_name = checkpoint.split("/")[-1]

        dataset, tokenizer = get_tokenized_mnli(checkpoint, return_tokenizer=True)
        eval_fn = make_eval_fn(dataset, tokenizer, DEVICE, eval_split="validation_matched")

        results = run_inference_benchmark(
            model_name,
            base_dir=Path.home(),
            device=DEVICE,
            eval_fn=eval_fn,
        )
        all_results.extend(results)

    save_and_print_results(all_results)


if __name__ == "__main__":
    main()