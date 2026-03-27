import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# 1. Setup Checkpoints (Switching to V3-Base as requested)
checkpoint = "microsoft/deberta-v3-base"
dataset_name = "imdb"

# 2. Device Check
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

# 3. Load Dataset and Tokenizer
# Using standard load_dataset to bypass custom tool issues
dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    # DeBERTa-v3 is powerful, but max_length 512 is the standard limit
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator handles dynamic padding for batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Load Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.config.problem_type = "single_label_classification"
model.to(DEVICE)

# 5. Metrics (Accuracy)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta-imdb-finetuned",
    learning_rate=2e-5,            # Standard for DeBERTa
    per_device_train_batch_size=8, # Adjust based on your VRAM (8-16 is good)
    per_device_eval_batch_size=8,
    num_train_epochs=1,            # IMDB usually converges in 1-2 epochs
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                     # Highly recommended for speed/memory
    logging_steps=100,
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Start Training
print("Starting training...")
trainer.train()

# 9. Save the Final Model
trainer.save_model("./final_deberta_imdb_model")
print("Done!")