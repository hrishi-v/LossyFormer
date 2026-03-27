import torch
from tqdm.auto import tqdm
from collections.abc import Mapping
from .HeadProfiler import HeadProfiler
import torch.nn.functional as F


def instrument_model(model):
    """
    Replaces FX graph instrumentation with standard PyTorch forward hooks.
    This works directly on the Hugging Face BertSelfAttention module.
    """
    modules = {}
    handles = []

    base = getattr(model, "base_model", model)
    if hasattr(base, "bert"):
        base = base.bert
    elif hasattr(base, "roberta"):
        base = base.roberta

    layers = getattr(base.encoder, "layer", [])
    original_num_heads = getattr(base.config, "num_attention_heads", 12)

    for i, layer in enumerate(layers):
        attn_module = layer.attention.self
        num_heads = attn_module.num_attention_heads
        head_size = attn_module.attention_head_size

        pruned_heads = getattr(attn_module, "pruned_heads", set())
        active_heads = [h for h in range(original_num_heads) if h not in pruned_heads]

        profiler = HeadProfiler(num_heads, head_size, active_heads)
        modules[i] = profiler

        def fwd_hook(module, inputs, output, prof=profiler):
            if not prof.collecting:
                return output

            context_layer = output[0] if isinstance(output, tuple) else output
            batch_size = context_layer.size(0)

            if context_layer.requires_grad:
                act = context_layer.detach()

                def bwd_hook(grad):
                    with torch.no_grad():
                        act_reshaped = act.view(
                            batch_size, -1, prof.num_heads, prof.head_size
                        )
                        grad_reshaped = grad.view(
                            batch_size, -1, prof.num_heads, prof.head_size
                        )

                        score = (grad_reshaped * act_reshaped).abs()
                        head_imp = score.mean(dim=(1, 3)).sum(dim=0).cpu()
                        prof.imp_scores.append((head_imp, batch_size))

                context_layer.register_hook(bwd_hook)
            else:
                with torch.no_grad():
                    act_reshaped = context_layer.view(
                        batch_size, -1, prof.num_heads, prof.head_size
                    )
                    mag_sum = act_reshaped.norm(dim=-1).mean(dim=1).sum(dim=0).cpu()
                    prof.imp_scores.append((mag_sum, batch_size))
            return output

        handle = attn_module.register_forward_hook(fwd_hook)
        handles.append(handle)

    return modules, handles


def remove_instrumentation(handles):
    """Cleans up hooks after calibration."""
    for handle in handles:
        handle.remove()


@torch.no_grad()
def calibrate_with_survival(
    model, modules, loader, thresholds=[0.3], device="cuda", n_batches=100
):
    """Calibrates head importance while tracking layer execution probability (P_e)."""
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    model.eval().to(device)
    for m in modules.values():
        m.collecting = True
        m.imp_scores = []

    layer_survival_counts = None
    total_samples = 0

    pbar = tqdm(
        loader, total=min(len(loader), n_batches), desc="Calibrating (P_e)", leave=False
    )
    for i, b in enumerate(pbar):
        if i >= n_batches:
            break

        inputs = {}
        if isinstance(b, Mapping):
            inputs = {k: v.to(device) for k, v in b.items() if k != "labels"}
        else:
            inputs = {"input_ids": b[0].to(device), "attention_mask": b[1].to(device)}

        outputs = model(**inputs, output_all_logits=True)
        all_logits = outputs["logits"]

        batch_size = all_logits[0].size(0)
        num_layers = len(all_logits)

        if layer_survival_counts is None:
            layer_survival_counts = torch.zeros(num_layers, device=device)

        logits_stack = torch.stack(all_logits, dim=1)
        probs = F.softmax(logits_stack, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        batch_counts_sum = torch.zeros(num_layers, device=device)

        for th in thresholds:
            exited = entropy <= th
            previously_exited = torch.cat(
                [
                    torch.zeros(batch_size, 1, device=device, dtype=torch.bool),
                    exited[:, :-1],
                ],
                dim=1,
            )
            prev_exit_cum = previously_exited.long().cumsum(dim=1)
            reached_mask = prev_exit_cum == 0
            batch_counts_sum += reached_mask.float().sum(dim=0)

        layer_survival_counts += batch_counts_sum
        total_samples += batch_size

    for m in modules.values():
        m.collecting = False

    if total_samples > 0:
        survival_probs = (
            (layer_survival_counts / (total_samples * len(thresholds))).cpu().tolist()
        )
    else:
        survival_probs = (
            [1.0] * len(layer_survival_counts)
            if layer_survival_counts is not None
            else []
        )

    return survival_probs


def decide_heads_to_prune(imp_modules, survival_probs, keep_ratio=0.5):
    """
    Ranks heads by Expected Importance and returns a dictionary of heads to PRUNE.
    Ensures exactly keep_ratio of heads are kept while guaranteeing 1 head per layer.
    """
    all_heads = []

    for i, mod_id in enumerate(sorted(imp_modules.keys())):
        mod = imp_modules[mod_id]
        scores = mod.get_scores()
        if scores is None:
            continue

        p_e = survival_probs[i] if (survival_probs and i < len(survival_probs)) else 1.0

        for rel_h, s in enumerate(scores):
            abs_h = mod.active_heads[rel_h]
            weighted = s * p_e if p_e > 0 else 0.0
            all_heads.append((mod_id, abs_h, weighted))

    if not all_heads:
        return {}

    total = len(all_heads)

    minimum_heads_to_keep = int(round(total * keep_ratio))

    if minimum_heads_to_keep == total:
        minimum_heads_to_keep = (
            total - 1
        )  # Ensure at least one head is pruned if keep ratio is too high

    n_keep = max(1, minimum_heads_to_keep)

    forced_kept = []
    layers = set(layer for layer, _, _ in all_heads)
    for layer in layers:
        best = max((x for x in all_heads if x[0] == layer), key=lambda x: x[2])
        forced_kept.append(best)

    remaining_to_keep = n_keep - len(forced_kept)

    if remaining_to_keep > 0:
        forced_set = set((l, h) for l, h, _ in forced_kept)
        remaining_heads = [x for x in all_heads if (x[0], x[1]) not in forced_set]
        remaining_heads.sort(key=lambda x: x[2], reverse=True)
        kept = forced_kept + remaining_heads[:remaining_to_keep]
    else:
        kept = forced_kept

    kept_set = set((l, h) for l, h, _ in kept)
    heads_to_prune = {}

    for layer, head, _ in all_heads:
        if (layer, head) not in kept_set:
            heads_to_prune.setdefault(layer, []).append(head)

    return heads_to_prune


def prune_heads_pass(model, heads_to_prune):
    """Uses native Hugging Face model.prune_heads()"""
    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "bert"):
        base_model = base_model.bert
    elif hasattr(base_model, "roberta"):
        base_model = base_model.roberta

    base_model.prune_heads(heads_to_prune)
    return model
