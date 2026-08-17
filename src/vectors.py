"""Construction of concept injection directions (Interventions 1 and 2)."""

import torch

from src.model_utils import last_token_hidden_state


def unit_norm(v, eps=1e-8):
    return v / (v.norm() + eps)


def single_vector(model, tokenizer, layer_module_path, concept_prompt, neutral_prompt, device="cuda"):
    """Intervention 1: v = activation(concept) - activation(neutral), unit-normalized."""
    h_concept = last_token_hidden_state(model, tokenizer, layer_module_path, concept_prompt, device)
    h_neutral = last_token_hidden_state(model, tokenizer, layer_module_path, neutral_prompt, device)
    v = h_concept - h_neutral
    return unit_norm(v), h_concept, h_neutral


def multi_example_contrastive_vector(model, tokenizer, layer_module_path, paraphrase_pairs, device="cuda"):
    """Intervention 2: for each (concept_i, neutral_i) pair compute a unit-normalized
    diff vector, then average across pairs. The mean is itself renormalized to unit
    norm so alpha scaling is comparable across intervention types.
    """
    unit_vecs = []
    for pair in paraphrase_pairs:
        h_c = last_token_hidden_state(model, tokenizer, layer_module_path, pair["concept"], device)
        h_n = last_token_hidden_state(model, tokenizer, layer_module_path, pair["neutral"], device)
        v_i = unit_norm(h_c - h_n)
        unit_vecs.append(v_i)
    stacked = torch.stack(unit_vecs, dim=0)
    mean_v = stacked.mean(dim=0)
    return unit_norm(mean_v), stacked
