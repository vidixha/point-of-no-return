"""Shared experiment utilities used by the v6 (final) experiment runner
and the Modal GPU app (scripts/modal_app.py)."""


def measure_activation_norm(model, tokenizer, layer_module_path, sample_texts, device):
    """Mean L2 norm of the residual stream at layer_module_path, last-token
    position, across a handful of representative prompts. Used to set alpha
    in physically meaningful units rather than an arbitrary raw scale."""
    from src.model_utils import last_token_hidden_state

    norms = []
    for t in sample_texts:
        h = last_token_hidden_state(model, tokenizer, layer_module_path, t, device=device)
        norms.append(h.norm().item())
    return sum(norms) / len(norms)
