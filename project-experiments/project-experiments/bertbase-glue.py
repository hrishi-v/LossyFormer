checkpoint = "bert-base-uncased"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "glue"
dataset_config = "mnli"

import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("Using CUDA device")
    print("Device name:", props.name)
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total GPU memory: {props.total_memory / 1024**3:.2f} GB")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available — using CPU")

print("DEVICE =", DEVICE)

from datasets import load_dataset
from transformers import AutoTokenizer

raw = load_dataset(dataset_name, dataset_config)
raw = raw.filter(lambda x: x["label"] >= 0)

# MNLI has validation_matched and validation_mismatched but no test labels
# Use validation_matched as the test split for evaluation
raw["test"] = raw["validation_matched"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
dataset = raw.map(
    lambda x: tokenizer(x["premise"], x["hypothesis"], truncation=True),
    batched=True,
)

from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=3,
)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

from chop.passes.module import report_trainable_parameters_analysis_pass

_, _ = report_trainable_parameters_analysis_pass(mg.model)

for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False

from chop.tools import get_trainer

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)

eval_results = trainer.evaluate()
print(eval_results)
print(f"Pre-training accuracy: {eval_results['eval_accuracy']}")

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
print(f"Post-training accuracy: {eval_results['eval_accuracy']}")

mg.export("/vol/bitbucket/ug22/adls-data/models/bert-base-glue-mnli-baseline")
