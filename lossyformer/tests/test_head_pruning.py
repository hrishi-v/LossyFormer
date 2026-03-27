import pytest
import torch
import torch.nn as nn
from lossyformer.pruning import HeadProfiler
from lossyformer.pruning import (
    instrument_model,
    remove_instrumentation,
    decide_heads_to_prune,
    prune_heads_pass,
)
from transformers import BertConfig, BertModel, BertForSequenceClassification

# Helpers
TINY_CFG = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    vocab_size=100,
    max_position_embeddings=32,
)


def _tiny_bert(**overrides):
    cfg = BertConfig(**{**TINY_CFG, **overrides})
    return BertModel(cfg)


def _tiny_classifier(num_labels=3, **overrides):
    cfg = BertConfig(**{**TINY_CFG, "num_labels": num_labels, **overrides})
    return BertForSequenceClassification(cfg)


def _dummy_inputs(batch=2, seq_len=8):
    ids = torch.randint(0, TINY_CFG["vocab_size"], (batch, seq_len))
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask}


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _fake_profilers(layer_scores):
    """Build profiler dict with pre-loaded scores.
    layer_scores: {layer_id: [score_per_head, ...]}
    """
    mods = {}
    for lid, scores in layer_scores.items():
        n = len(scores)
        prof = HeadProfiler(num_heads=n, head_size=16, active_heads=list(range(n)))
        prof.imp_scores.append((torch.tensor(scores, dtype=torch.float), 1))
        mods[lid] = prof
    return mods


# tests
def test_profiler_returns_none_when_no_scores_collected():
    prof = HeadProfiler(num_heads=4, head_size=16, active_heads=[0, 1, 2, 3])
    assert prof.get_scores() is None


def test_profiler_returns_none_when_total_samples_is_zero():
    prof = HeadProfiler(num_heads=2, head_size=8, active_heads=[0, 1])
    prof.imp_scores.append((torch.tensor([1.0, 1.0]), 0))
    assert prof.get_scores() is None


def test_profiler_computes_per_head_average_over_single_batch():
    prof = HeadProfiler(num_heads=3, head_size=16, active_heads=[0, 1, 2])
    prof.imp_scores.append((torch.tensor([2.0, 4.0, 6.0]), 2))
    scores = prof.get_scores()
    assert len(scores) == 3
    assert pytest.approx(scores, abs=1e-5) == [1.0, 2.0, 3.0]


def test_profiler_averages_across_multiple_batches():
    prof = HeadProfiler(num_heads=2, head_size=8, active_heads=[0, 1])
    prof.imp_scores.append((torch.tensor([4.0, 8.0]), 2))
    prof.imp_scores.append((torch.tensor([6.0, 2.0]), 2))
    scores = prof.get_scores()
    # sum=[10, 10], samples=4 → [2.5, 2.5]
    assert pytest.approx(scores, abs=1e-5) == [2.5, 2.5]


def test_profiler_collecting_flag_defaults_to_false():
    prof = HeadProfiler(num_heads=2, head_size=8, active_heads=[0, 1])
    assert prof.collecting is False


def test_instrument_creates_one_profiler_per_layer():
    model = _tiny_bert(num_hidden_layers=3)
    modules, handles = instrument_model(model)
    remove_instrumentation(handles)
    assert len(modules) == 3
    assert len(handles) == 3


def test_instrument_profiler_head_count_matches_config():
    model = _tiny_bert(num_attention_heads=4)
    modules, handles = instrument_model(model)
    remove_instrumentation(handles)
    for prof in modules.values():
        assert prof.num_heads == 4
        assert len(prof.active_heads) == 4


def test_instrument_computes_correct_head_size():
    model = _tiny_bert(hidden_size=64, num_attention_heads=4)
    modules, handles = instrument_model(model)
    remove_instrumentation(handles)
    for prof in modules.values():
        assert prof.head_size == 16  # 64 / 4


def test_instrument_finds_layers_through_classifier_wrapper():
    model = _tiny_classifier()
    modules, handles = instrument_model(model)
    remove_instrumentation(handles)
    assert len(modules) == 2


def test_instrument_returns_empty_for_model_without_encoder_layers():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList()
            self.config = type("C", (), {"num_attention_heads": 4})()

    modules, handles = instrument_model(Dummy())
    assert len(modules) == 0
    assert len(handles) == 0


def test_remove_instrumentation_prevents_further_collection():
    model = _tiny_bert()
    modules, handles = instrument_model(model)
    remove_instrumentation(handles)
    for prof in modules.values():
        prof.collecting = True
    with torch.no_grad():
        model(**_dummy_inputs())
    for prof in modules.values():
        assert prof.get_scores() is None  # hooks gone, nothing collected


def test_taylor_scores_collected_when_gradients_flow():
    model = _tiny_classifier()
    modules, handles = instrument_model(model)
    for prof in modules.values():
        prof.collecting = True
    inputs = _dummy_inputs()
    labels = torch.zeros(inputs["input_ids"].size(0), dtype=torch.long)
    outputs = model(**inputs, labels=labels)
    outputs.loss.backward()
    remove_instrumentation(handles)
    for prof in modules.values():
        scores = prof.get_scores()
        assert scores is not None
        assert len(scores) == prof.num_heads


def test_multiple_forward_passes_accumulate_score_entries():
    model = _tiny_bert()
    modules, handles = instrument_model(model)
    for prof in modules.values():
        prof.collecting = True
    with torch.no_grad():
        model(**_dummy_inputs())
        model(**_dummy_inputs())
        model(**_dummy_inputs())
    remove_instrumentation(handles)
    for prof in modules.values():
        assert len(prof.imp_scores) == 3


def test_decide_prunes_exactly_half_at_keep_ratio_half():
    mods = _fake_profilers({0: [1, 2, 3, 4], 1: [4, 3, 2, 1]})
    pruned = decide_heads_to_prune(mods, [1.0, 1.0], keep_ratio=0.5)
    total_pruned = sum(len(v) for v in pruned.values())
    assert total_pruned == 4  # 8 total, keep 4, prune 4


def test_decide_guarantees_at_least_one_head_per_layer():
    # Layer 0 has uniformly low scores, layer 1 much higher
    mods = _fake_profilers({0: [0.01, 0.01, 0.01, 0.01], 1: [10, 10, 10, 10]})
    pruned = decide_heads_to_prune(mods, [1.0, 1.0], keep_ratio=0.25)
    pruned_in_layer0 = len(pruned.get(0, []))
    assert pruned_in_layer0 <= 3  # at least 1 kept


def test_decide_returns_empty_dict_for_empty_input():
    assert decide_heads_to_prune({}, []) == {}


def test_decide_skips_profilers_with_no_scores():
    prof = HeadProfiler(num_heads=4, head_size=16, active_heads=[0, 1, 2, 3])
    assert decide_heads_to_prune({0: prof}, [1.0]) == {}


def test_decide_survival_prob_zero_deprioritises_layer():
    mods = _fake_profilers({0: [10, 10, 10, 10], 1: [1, 1, 1, 1]})
    pruned = decide_heads_to_prune(mods, [0.0, 1.0], keep_ratio=0.5)
    # Layer 0 has weighted score 0, so more of its heads should be pruned
    assert len(pruned.get(0, [])) >= len(pruned.get(1, []))


def test_decide_keep_ratio_one_prunes_one_head():
    mods = _fake_profilers({0: [1, 2], 1: [3, 4]})
    pruned = decide_heads_to_prune(mods, [1.0, 1.0], keep_ratio=1.0)
    assert sum(len(v) for v in pruned.values()) == 1


def test_decide_returns_valid_head_indices_with_no_duplicates():
    mods = _fake_profilers({0: [5, 1, 3, 2], 1: [2, 4, 1, 3]})
    pruned = decide_heads_to_prune(mods, [1.0, 1.0], keep_ratio=0.5)
    for layer_id, heads in pruned.items():
        assert all(0 <= h < 4 for h in heads)
        assert len(heads) == len(set(heads))


def test_prune_reduces_parameter_count():
    model = _tiny_bert()
    before = _param_count(model)
    prune_heads_pass(model, {0: [0, 1], 1: [2, 3]})
    assert _param_count(model) < before


def test_model_runs_forward_pass_after_pruning():
    model = _tiny_bert()
    prune_heads_pass(model, {0: [0], 1: [1]})
    with torch.no_grad():
        out = model(**_dummy_inputs())
    assert out.last_hidden_state.shape[0] == 2


def test_prune_updates_num_heads_in_attention_module():
    model = _tiny_bert(num_attention_heads=4)
    prune_heads_pass(model, {0: [0, 1]})
    attn = model.encoder.layer[0].attention.self
    assert attn.num_attention_heads == 2


def test_prune_empty_dict_changes_nothing():
    model = _tiny_bert()
    before = _param_count(model)
    prune_heads_pass(model, {})
    assert _param_count(model) == before


def test_prune_works_on_classifier_model():
    model = _tiny_classifier()
    before = _param_count(model)
    prune_heads_pass(model, {0: [0, 1]})
    assert _param_count(model) < before


def test_prune_only_affects_requested_layer():
    model = _tiny_bert(num_attention_heads=4, num_hidden_layers=2, hidden_size=64)
    layer0_attn = model.encoder.layer[0].attention.self
    layer1_attn = model.encoder.layer[1].attention.self

    layer1_q_shape_before = layer1_attn.query.weight.shape
    layer1_k_shape_before = layer1_attn.key.weight.shape
    layer1_v_shape_before = layer1_attn.value.weight.shape

    # Only prune layer 0
    prune_heads_pass(model, {0: [0, 1]})

    # Layer 0 should have fewer heads → smaller Q/K/V output dim
    assert layer0_attn.num_attention_heads == 2
    assert layer0_attn.query.weight.shape[0] < 64

    # Layer 1 should be completely unchanged
    assert layer1_attn.num_attention_heads == 4
    assert layer1_attn.query.weight.shape == layer1_q_shape_before
    assert layer1_attn.key.weight.shape == layer1_k_shape_before
    assert layer1_attn.value.weight.shape == layer1_v_shape_before


def _run_one_prune_cycle(model, keep_ratio=0.5):
    modules, handles = instrument_model(model)
    for prof in modules.values():
        prof.collecting = True
    with torch.no_grad():
        for _ in range(3):
            model(**_dummy_inputs())
    for prof in modules.values():
        prof.collecting = False
    remove_instrumentation(handles)

    num_layers = len(modules)
    survival = [1.0] * num_layers
    heads_to_prune = decide_heads_to_prune(modules, survival, keep_ratio=keep_ratio)
    prune_heads_pass(model, heads_to_prune)
    return heads_to_prune


def test_single_prune_cycle_reduces_params_and_model_runs():
    model = _tiny_bert()
    before = _param_count(model)
    heads_to_prune = _run_one_prune_cycle(model, keep_ratio=0.5)
    assert len(heads_to_prune) > 0
    assert _param_count(model) < before
    with torch.no_grad():
        out = model(**_dummy_inputs())
    assert out.last_hidden_state.shape[0] == 2


def test_two_prune_cycles_reduce_params_further():
    model = _tiny_bert(num_attention_heads=4, num_hidden_layers=2)
    before = _param_count(model)

    _run_one_prune_cycle(model, keep_ratio=0.75)
    after_first = _param_count(model)
    assert after_first < before

    _run_one_prune_cycle(model, keep_ratio=0.75)
    after_second = _param_count(model)
    assert after_second < after_first

    with torch.no_grad():
        out = model(**_dummy_inputs())
    assert out.last_hidden_state is not None


def test_correct_heads_removed_across_two_iterations():
    """Prune absolute head 1 in round 1, then absolute head 3 in round 2.
    Verify the surviving weights are exactly original heads 0 and 2."""
    model = _tiny_bert(num_attention_heads=4, hidden_size=64, num_hidden_layers=1)
    attn = model.encoder.layer[0].attention.self
    head_size = 16

    q_original = attn.query.weight.data.clone()
    original_slices = [
        q_original[i * head_size : (i + 1) * head_size, :] for i in range(4)
    ]

    # Round 1: prune absolute head 1
    prune_heads_pass(model, {0: [1]})
    assert attn.query.weight.shape[0] == 3 * head_size

    # Round 2: prune absolute head 3 (not position 1)
    prune_heads_pass(model, {0: [3]})
    assert attn.query.weight.shape[0] == 2 * head_size

    q_final = attn.query.weight.data
    for new_idx, original_idx in enumerate([0, 2]):
        surviving = q_final[new_idx * head_size : (new_idx + 1) * head_size, :]
        assert torch.equal(surviving, original_slices[original_idx]), (
            f"Position {new_idx} should be original head {original_idx}"
        )


def test_decided_head_indices_are_valid_for_current_model():
    model = _tiny_bert(num_attention_heads=4, num_hidden_layers=2)

    prune_heads_pass(model, {0: [0], 1: [3]})

    modules, handles = instrument_model(model)
    for prof in modules.values():
        prof.collecting = True
    with torch.no_grad():
        model(**_dummy_inputs())
    for prof in modules.values():
        prof.collecting = False
    remove_instrumentation(handles)

    heads_to_prune = decide_heads_to_prune(modules, [1.0, 1.0], keep_ratio=0.5)

    for layer_id, heads in heads_to_prune.items():
        current_num_heads = modules[layer_id].num_heads
        for h in heads:
            assert h < current_num_heads, (
                f"Layer {layer_id}: head index {h} >= current head count {current_num_heads}"
            )
