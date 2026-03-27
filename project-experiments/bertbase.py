checkpoint = "bert-base-uncased"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

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

from chop.tools import get_tokenized_dataset
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
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
print(f"Pre-training accuracy: {eval_results['eval_accuracy']}")
trainer.train()
eval_results = trainer.evaluate()
print(f"Post-training accuracy: {eval_results['eval_accuracy']}")

mg.export("/vol/bitbucket/ug22/adls-data/models/bert-base-imdb-baseline")
