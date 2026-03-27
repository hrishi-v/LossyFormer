# LossyFormer

<p align="center">
  <img src="LossyFormerLogo.png" width="400" alt="Description">
</p>

This forms a suite of tools developed as part of the Spring 2026 ADLS module. It is aimed at optimising several transformer models for latency and throughput, whilst accepting minor, controlled accuracy drops. It is tested and working for BERT-Base, BERT-Tiny and RoBERTa.

## Usage

For an example of how to use the tool, see`lf_tests/example.py`. This demonstrates how to instantiate the class and call `lossy.fit()` on transformer models. We do this with a BERT-Tiny model, on the IMDB dataset, permitting an accuracy loss of 1.5%. Note that this is primarily provided as a sanity check that the tool works end-to-end.

```sh
uv sync
source .venv/bin/activate
python lf_tests/example.py # Executes the lossy.fit() method
```

**TL:DR Test**

To quickly run all unit test then execute LossyFormer on the compatible models, use the following commands.

```sh
chmod +x test_all.sh
./test_all.sh
```

**Individual tests**

Defined below are the commands to run LossyFormer on BERT-Tiny, BERT-Base and RoBERTa-Base.

```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model bert-tiny # Executes the lossy.fit() method on the IMDB-trained BERT-Tiny model
```

```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model bert-base # Executes the lossy.fit() method on the MNLI-trained BERT-Base model
```

```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model roberta # Executes the lossy.fit() method on the MNLI-trained RoBERTa model
```

Bear in mind, this will take a *long* while, so might be best to see how it works and interrupt it :). The `step_keep_ratio` is set to 0.9 by default but can be changed to control the aggressiveness of head pruning. While a smaller `step_keep_ratio` will converge faster, it may prune too aggressively per iteration to find the best tradeoff between accuracy drop and latency for a model.

**Unit Testing**

To run all the unit tests, run `pytest`.

## Future Goals

- Develop and test the LossyFormer utility for the ViT model.
- Reflect on the GeLU to ReLU swap, which is expected to offer better latency and throughput, but for custom architectures built on FPGA or ASIC. Omitted due to GPU performance being unaffected.


**Contributors:**

- Dhyey Trivedi
- Hrishikesh Venkatesh
- Neil Radhu
- Utsav Goel