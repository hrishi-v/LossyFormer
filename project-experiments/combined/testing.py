import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset
from module import LossyFormer, eval_accuracy, eval_speed, MaseGraph
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# checkpoint_name = "roberta-base"
# checkpoint_name = "bert-base-uncased"
checkpoint_name = "prajjwal1/bert-tiny"

# baseline_checkpoint = "/vol/bitbucket/ug22/adls-data/models/roberta-base-glue-mnli-baseline"
# Based on ls output: /vol/bitbucket/ug22/adls-data/models/bert-base-glue-mnli-baseline.pt
# baseline_checkpoint = "/vol/bitbucket/ug22/adls-data/models/bert-base-glue-mnli-baseline"
baseline_checkpoint = "/vol/bitbucket/ug22/adls-data/models/bert-tiny-glue-mnli-baseline"
# baseline_checkpoint = "/vol/bitbucket/hv122/adls-data/roberta-base-glue-mnli-baseline"
print(f"Loading Dataset (GLUE MNLI) with checkpoint {checkpoint_name}...")
raw = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)

tokenized_datasets = raw.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def make_loader(split_name, batch_size=64):
    ds = tokenized_datasets[split_name]
    keep_cols = ["input_ids", "attention_mask", "label"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols).rename_column("label", "labels").with_format("torch")
    return DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)

train_loader = make_loader("train", batch_size=32)
test_loader = make_loader("validation_matched", batch_size=64)

print("\n" + "="*50)
print("PHASE 1: INITIAL BASELINE EVALUATION")
print("="*50)
base_mg = MaseGraph.from_checkpoint(baseline_checkpoint)
base_acc = eval_accuracy(base_mg.model, test_loader, DEVICE)
base_tput, _, base_vram = eval_speed(base_mg.model, test_loader, DEVICE)

print(f"Baseline Accuracy: {base_acc*100:.2f}%")
print(f"Baseline Throughput: {base_tput:.2f} samples/sec")
print(f"Baseline Inference VRAM: {base_vram:.2f} MB")

del base_mg
torch.cuda.empty_cache()

print("\n" + "="*50)
print("PHASE 2: LOSSYFORMER PRUNING & LORA HEALING")
print("="*50)
lossy = LossyFormer(allowed_accuracy_loss=0.03, device=DEVICE)

# If Mase saved the entire FX graph object instead of a pure state dict:
if hasattr(state_dict, "state_dict"):
    state_dict = state_dict.state_dict()
    
hf_model.load_state_dict(state_dict, strict=False)

lossy = LossyFormer(allowed_accuracy_loss=0.1, device=DEVICE)

print("Starting LossyFormer fit process...")
# Pass the instantiated model instead of the path
model = lossy.fit(
    hf_model, 
    train_loader, 
    test_loader, 
    # max_ft_steps=30 
)

print("\nFinal Evaluation of Early Exit Model:")
acc = eval_accuracy(model, test_loader, DEVICE)
print(f"Final Accuracy: {acc*100:.2f}%")

throughput, latency = eval_speed(model, test_loader, DEVICE)
print(f"Final Throughput: {throughput:.2f} samples/sec")
print(f"Final Latency: {latency*1000:.2f} ms/batch")