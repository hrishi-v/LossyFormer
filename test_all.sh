#!/bin/bash
uv sync
source .venv/bin/activate

# Run unit tests
echo "Running Unit Tests..."
pytest

# Run the single-pass LossyFormer pipeline (Targeting a 3% accuracy drop)
echo "Starting LossyFormer TL;DR Pipeline (3% Target Drop)..."

echo "-> Running BERT-Tiny on IMDB..."
python lf_tests/lf-testing-light.py --model bert-tiny --target_drop 0.03

echo "-> Running BERT-Base on MNLI..."
python lf_tests/lf-testing-light.py --model bert-base --target_drop 0.03

echo "-> Running RoBERTa-Base on MNLI..."
python lf_tests/lf-testing-light.py --model roberta --target_drop 0.03

echo "All tests complete!"
