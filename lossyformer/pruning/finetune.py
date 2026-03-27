import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections.abc import Mapping
from peft import LoraConfig, get_peft_model

def fine_tune_lora(
    model,
    train_loader,
    eval_loader,
    epochs=None,
    max_steps=None,
    lr=3e-4,
    device="cuda",
):
    """Fine-tune using LoRA: freeze base weights, train low-rank adapters, then merge."""
    
    # PEFT natively supports substring matching for Hugging Face models
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier", "pooler", "score"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.train().to(device)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=lr)
    
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler(device_type)

    steps = 0
    total_epochs = epochs if epochs is not None else 1
    is_early_exit = hasattr(model, "evaluate_confidence") or hasattr(model.base_model, "evaluate_confidence")

    for ep in range(total_epochs):
        losses = []
        pbar = tqdm(train_loader, desc=f"LoRA fine-tuning (Epoch {ep + 1})", leave=False)
        for b in pbar:
            with torch.autocast(device_type=device_type):
                if isinstance(b, Mapping):
                    b = {k: v.to(device) for k, v in b.items()}
                    labels = b.get("labels")

                    if is_early_exit:
                        outputs = peft_model(**b, output_all_logits=True)
                        if "logits" in outputs and isinstance(outputs["logits"], list):
                            loss = sum(F.cross_entropy(logits, labels) for logits in outputs["logits"])
                        else:
                            loss = F.cross_entropy(outputs["logits"], labels) if "logits" in outputs else outputs["loss"]
                    else:
                        outputs = peft_model(**b)
                        loss = outputs["loss"] if isinstance(outputs, Mapping) and "loss" in outputs else outputs.loss

                else:
                    inputs, targets = b[0].to(device), b[1].to(device)
                    if is_early_exit:
                        outputs = peft_model(inputs, attention_mask=None, output_all_logits=True)
                        if "logits" in outputs and isinstance(outputs["logits"], list):
                            loss = sum(F.cross_entropy(logits, targets) for logits in outputs["logits"])
                        else:
                            loss = F.cross_entropy(outputs["logits"], targets) if "logits" in outputs else F.cross_entropy(outputs, targets)
                    else:
                        outputs = peft_model(inputs)
                        loss = F.cross_entropy(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses[-10:]))

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        if max_steps is not None and steps >= max_steps:
            break

    print("  Merging LoRA adapters back into base model...")
    return peft_model.merge_and_unload()