# LossyFormer

<p align="center">
  <img src="LossyFormerLogo.png" width="400" alt="Description">
</p>

This forms a suite of tools developed as part of the Spring 2026 ADLS module. It is aimed at optimising several transformer models for latency and throughput, whilst accepting minor, controlled accuracy drops. It is tested and working for BERT-Base, BERT-Tiny and RoBERTa.

For further information on the intracacies of what our tool does and support for models, check [here](lossyformer/docs.md).

## Usage

For an example of how to use the tool, see`lf_tests/example.py`. This demonstrates how to instantiate the class and call `lossy.fit()` on transformer models. We do this with a BERT-Tiny model, on the IMDB dataset, permitting an accuracy loss of 1.5%. Note that this is primarily provided as a sanity check that the tool works end-to-end.


```sh
uv sync
source .venv/bin/activate
python lf_tests/example.py # Executes the lossy.fit() method
```

**TL:DR Test**

To quickly run all unit tests then execute a single LossyFormer pass at an accuracy drop of 3% on the compatible models, use the following commands. This is aimed at being primarily demonstrative, so we only use a single finetuning iteration and one pruning step.

Take the finetuned models from this [SharePoint link](https://imperiallondon-my.sharepoint.com/:u:/r/personal/hv122_ic_ac_uk/Documents/models.zip?csf=1&web=1&e=iBeXvS).

Use the models here: 

```sh
chmod +x test_all.sh
./test_all.sh
```

**Individual Sweeps**

Defined below are the commands to run LossyFormer search sweeps with varying acceptable accuracy drops on BERT-Tiny, BERT-Base and RoBERTa-Base. Bear in mind, this will take a *long* while, so might be best to see how it works and interrupt it :). This is because all three are programmed to run a sweep across different acceptable accuracy losses.

BERT-Tiny, trained on IMDB:
```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model bert-tiny # Executes the lossy.fit() method on the IMDB-trained BERT-Tiny model
```

BERT-Base, trained on MNLI:
```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model bert-base # Executes the lossy.fit() method on the MNLI-trained BERT-Base model
```

RoBERTa, trained on MNLI:
```sh
# Run the sync and source the .venv first!
python lf_tests/lf-testing.py --model roberta # Executes the lossy.fit() method on the MNLI-trained RoBERTa model
```

The `step_keep_ratio` is set to 0.9 by default but can be changed to control the aggressiveness of head pruning. While a smaller `step_keep_ratio` will converge faster, it may prune too aggressively per iteration to find the best tradeoff between accuracy drop and latency for a model.

**Unit Testing**

To run all the unit tests, run `pytest`.

## Future Goals

- Develop and test the LossyFormer utility for the ViT model.
- Reflect on the GeLU to ReLU swap, which is expected to offer better latency and throughput, but for custom architectures built on FPGA or ASIC. Omitted due to GPU performance being unaffected.


## Project Experiments

As part of this project, we conducted a wide range of experiments on our baseline models. These included evaluating the impact of different activation functions (e.g., replacing GeLU with ReLU) on both throughput and latency, exploring various approximations of Scaled Dot-Product Attention (SDPA), and applying attention head pruning based on different head importance metrics.

We also investigated early exit strategies, experimenting with multiple criteria such as Maximum Probability and Shannon Entropy, along with different methods for selecting optimal exit thresholds (e.g. Grid Seach and Bayesian Optimisation).

A subset of these experiments is available in the `project-experiments/` directory, alongside our final implementation.

**Contributors:**

- Dhyey Trivedi
- Hrishikesh Venkatesh
- Neil Radhu
- Utsav Goel
