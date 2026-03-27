import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from chop import MaseGraph
from chop.tools import get_tokenized_dataset
from transformers import DataCollatorWithPadding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CKPT = "/vol/bitbucket/ug22/adls-data/models/bert-base-imdb-baseline"
TOKENIZER_CKPT = "bert-base-uncased"

# === Data ===

dataset, tokenizer = get_tokenized_dataset(dataset="imdb", checkpoint=TOKENIZER_CKPT, return_tokenizer=True)
collator = DataCollatorWithPadding(tokenizer)

def make_loader(split, bs, shuffle=False):
    ds = dataset[split].remove_columns(["text", "token_type_ids"]).rename_column("label", "labels")
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collator)

train_loader = make_loader("train", 32, shuffle=True)
eval_loader = make_loader("test", 64)

# === Eval ===

@torch.no_grad()
def eval_accuracy(model, loader, device=DEVICE):
    model.eval().to(device)
    correct = sum(
        (model(**{k: v.to(device) for k, v in b.items()})["logits"].argmax(-1) == b["labels"].to(device)).sum().item()
        for b in loader
    )
    acc = correct / len(loader.dataset)
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

@torch.no_grad()
def eval_speed(model, loader, device="cuda", n=50, warmup=5):
    model.eval().to(device)
    compiled = torch.compile(model)
    batches = list(loader)[:warmup + n]
    for b in batches[:warmup]:
        compiled(**{k: v.to(device) for k, v in b.items()})
    if device == "cuda": torch.cuda.synchronize()
    samples, times = 0, []
    for b in batches[warmup:]:
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled(**{k: v.to(device) for k, v in b.items()})
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        samples += b["input_ids"].size(0)
    print(f"{samples} samples, {samples/sum(times):.0f} samples/sec, {np.mean(times)*1000:.1f} ms/batch")


class ImportanceSDPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = []
        self.collecting = True

    def forward(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        if self.collecting:
            with torch.no_grad():
                self.norms.append(out.norm(dim=-1).mean(dim=(0, 2)).cpu())
        return out

    def get_scores(self):
        return torch.stack(self.norms).mean(0).tolist() if self.norms else None


def instrument_sdpa_pass(mg):
    modules = {}
    for i, node in enumerate(n for n in list(mg.fx_graph.nodes)
                              if n.op == "call_function" and "scaled_dot_product_attention" in str(n.target)):
        name = f"_imp_sdpa_{i}"
        mod = ImportanceSDPA()
        mg.model.add_module(name, mod)
        modules[i] = mod
        with mg.fx_graph.inserting_before(node):
            new = mg.fx_graph.call_module(name, args=node.args, kwargs=node.kwargs)
            node.replace_all_uses_with(new)
            mg.fx_graph.erase_node(node)
    mg.fx_graph.lint()
    mg.model.recompile()
    return mg, modules


@torch.no_grad()
def calibrate(model, modules, loader, device=DEVICE, n_batches=100):
    model.eval().to(device)
    for m in modules.values(): m.collecting, m.norms = True, []
    for i, b in enumerate(loader):
        if i >= n_batches: break
        model(**{k: v.to(device) for k, v in b.items()})
    for m in modules.values(): m.collecting = False


def find_sdpa_contexts(mg):
    contexts = []
    for node in mg.fx_graph.nodes:
        if node.op != "call_function" or "scaled_dot_product_attention" not in str(node.target):
            continue
        ctx = {"sdpa_node": node, "qkv_linears": [], "view_add_nodes": [], "reshape_node": None, "output_dense": None}
        for arg in node.args[:3]:
            if arg.target == "permute":
                view_node = arg.args[0]
                if view_node.target == "view":
                    if view_node.args[0].op == "call_module":
                        ctx["qkv_linears"].append(view_node.args[0].target)
                    for a in view_node.args[1:]:
                        if hasattr(a, 'op') and a.op == "call_function" and "add" in str(a.target):
                            ctx["view_add_nodes"].append(a)
        for user in node.users:
            if user.target == "transpose":
                for ru in user.users:
                    if ru.target == "reshape":
                        ctx["reshape_node"] = ru
                        for du in ru.users:
                            if du.op == "call_module":
                                ctx["output_dense"] = du.target
        contexts.append(ctx)
    return contexts


def decide_heads_to_keep(imp_modules, keep_ratio=0.5):
    """
    Global pruning: rank ALL heads across ALL layers by L2 norm,
    keep the top keep_ratio fraction. Always keep at least 1 per layer.
    """
    # Collect all (layer, head, score) tuples
    all_heads = []
    for layer_idx, mod in imp_modules.items():
        scores = mod.get_scores()
        for h, s in enumerate(scores):
            all_heads.append((layer_idx, h, s))

    total = len(all_heads)
    n_keep = max(1, int(round(total * keep_ratio)))

    # Sort by score descending, take top n_keep
    all_heads.sort(key=lambda x: x[2], reverse=True)
    kept = all_heads[:n_keep]

    # Ensure at least 1 head per layer
    layers = set(layer for layer, _, _ in all_heads)
    for layer in layers:
        if not any(l == layer for l, _, _ in kept):
            # Add back the best head from this layer
            best = max((x for x in all_heads if x[0] == layer), key=lambda x: x[2])
            kept.append(best)

    # Build dict
    heads_to_keep = {}
    for layer, head, score in kept:
        heads_to_keep.setdefault(layer, []).append(head)

    # Fill in layers that aren't in the dict (keep all)
    num_heads_per_layer = {layer: len(mod.get_scores()) for layer, mod in imp_modules.items()}
    for layer, nh in num_heads_per_layer.items():
        if layer not in heads_to_keep:
            heads_to_keep[layer] = list(range(nh))

    return heads_to_keep


def prune_heads_pass(mg, heads_to_keep_per_layer):
    contexts = find_sdpa_contexts(mg)
    for layer_idx, ctx in enumerate(contexts):
        keep = sorted(heads_to_keep_per_layer.get(layer_idx, []))
        old_shape = None
        for add_node in ctx["view_add_nodes"]:
            for arg in add_node.args:
                if isinstance(arg, tuple) and len(arg) == 2:
                    old_shape = arg
                    break
            if old_shape:
                break
        if old_shape is None:
            continue
        num_heads, head_dim = old_shape
        if not keep: keep = list(range(num_heads))
        if len(keep) >= num_heads:
            continue
        new_num_heads = len(keep)
        new_hidden = new_num_heads * head_dim
        keep_idx = torch.tensor([i for h in keep for i in range(h * head_dim, (h + 1) * head_dim)])
        print(f"SDPA {layer_idx}: {num_heads}h -> {new_num_heads}h (keep {keep})")
        for target in ctx["qkv_linears"]:
            mod = mg.model.get_submodule(target)
            mod.weight = nn.Parameter(mod.weight.data[keep_idx])
            if mod.bias is not None: mod.bias = nn.Parameter(mod.bias.data[keep_idx])
            mod.out_features = new_hidden
        if ctx["output_dense"]:
            mod = mg.model.get_submodule(ctx["output_dense"])
            mod.weight = nn.Parameter(mod.weight.data[:, keep_idx])
            mod.in_features = new_hidden
        for add_node in ctx["view_add_nodes"]:
            new_args = list(add_node.args)
            for i, a in enumerate(new_args):
                if a == (num_heads, head_dim): new_args[i] = (new_num_heads, head_dim)
            add_node.args = tuple(new_args)
        if ctx["reshape_node"]:
            new_args = list(ctx["reshape_node"].args)
            if new_args[-1] == num_heads * head_dim:
                new_args[-1] = new_hidden
                ctx["reshape_node"].args = tuple(new_args)
    mg.fx_graph.lint()
    mg.model.recompile()
    return mg


def fine_tune(model, train_loader, eval_loader, epochs=1, lr=2e-5, device=DEVICE):
    model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        losses = []
        for b in tqdm(train_loader, desc=f"Epoch {ep+1}"):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b)["loss"]
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(loss.item())
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f}")
        eval_accuracy(model, eval_loader, device)
    return model


# === Main experiment ===

KEEP_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1]

# Baseline
print("=== BASELINE ===")
mg = MaseGraph.from_checkpoint(MODEL_CKPT)
baseline_acc = eval_accuracy(mg.model, eval_loader)
print("GPU:"); eval_speed(mg.model, eval_loader, "cuda")
print("CPU:"); eval_speed(mg.model, eval_loader, "cpu")

# Profile once
print("\n=== PROFILE ===")
mg, imp_modules = instrument_sdpa_pass(mg)
calibrate(mg.model, imp_modules, eval_loader, n_batches=100)
for i, mod in imp_modules.items():
    print(f"SDPA {i}: {['%.4f' % s for s in mod.get_scores()]}")

# Sweep
# In the sweep loop, also collect speed and params:
results = []
for ratio in KEEP_RATIOS:
    print(f"\n{'='*40}")
    print(f"KEEP RATIO = {ratio}")
    print(f"{'='*40}")

    heads_to_keep = decide_heads_to_keep(imp_modules, keep_ratio=ratio)
    for layer, heads in sorted(heads_to_keep.items()):
        print(f"  SDPA {layer}: keep heads {heads}")

    mg_p = MaseGraph.from_checkpoint(MODEL_CKPT)
    mg_p = prune_heads_pass(mg_p, heads_to_keep)
    pruned_acc = eval_accuracy(mg_p.model, eval_loader)

    if ratio < 1.0:
        mg_p.model = fine_tune(mg_p.model, train_loader, eval_loader, epochs=1)
        ft_acc = eval_accuracy(mg_p.model, eval_loader)
    else:
        ft_acc = pruned_acc

    # Collect speed
    import io, contextlib
    def get_speed(model, loader, device):
        model.eval().to(device)
        compiled = torch.compile(model)
        batches = list(loader)[:10]
        for b in batches[:5]:
            compiled(**{k: v.to(device) for k, v in b.items()})
        if device == "cuda": torch.cuda.synchronize()
        samples, times = 0, []
        for b in batches[5:]:
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            compiled(**{k: v.to(device) for k, v in b.items()})
            if device == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            samples += b["input_ids"].size(0)
        return samples / sum(times)

    gpu_throughput = get_speed(mg_p.model, eval_loader, "cuda")
    cpu_throughput = get_speed(mg_p.model, eval_loader, "cpu")
    params = sum(p.numel() for p in mg_p.model.parameters())

    results.append({
        "ratio": ratio, "pruned": pruned_acc, "finetuned": ft_acc,
        "gpu": gpu_throughput, "cpu": cpu_throughput, "params": params,
    })

# Final summary
print(f"\n{'='*80}")
print(f"{'Ratio':>6} {'Pruned':>8} {'Finetuned':>9} {'vs Base':>8} {'Params':>10} {'GPU s/s':>9} {'CPU s/s':>9}")
print(f"{'='*80}")
for r in results:
    print(f"{r['ratio']:>6.0%} {r['pruned']*100:>7.2f}% {r['finetuned']*100:>8.2f}% "
          f"{(r['finetuned']-baseline_acc)*100:>+7.2f}% {r['params']:>10,} "
          f"{r['gpu']:>8.0f} {r['cpu']:>8.0f}")
