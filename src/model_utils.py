"""Model/tokenizer loading and residual-stream module lookup helpers."""

import functools

import torch


def get_module_by_path(model, dotted_path):
    """Resolve 'model.layers.18' -> the actual submodule object."""
    module = model
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def load_model_and_tokenizer(hf_id, revision, dtype="bfloat16", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision, dtype=torch_dtype
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, user_text, chat_template_kwargs=None):
    """Wrap a raw user message in the model's chat template."""
    messages = [{"role": "user", "content": user_text}]
    kwargs = dict(chat_template_kwargs or {})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def last_token_hidden_state(model, tokenizer, layer_module_path, text, device="cuda"):
    """Run a single forward pass and return the residual stream at the given
    layer's output, at the last (non-padding) token position. Shape: (hidden_size,)
    """
    module = get_module_by_path(model, layer_module_path)
    captured = {}

    def hook(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["h"] = hidden.detach()

    handle = module.register_forward_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    h = captured["h"]  # (batch=1, seq, hidden)
    return h[0, -1, :].float()
