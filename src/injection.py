"""Activation-addition injection: add alpha * v to the residual stream at a
chosen layer, at every token position, for every forward pass during a
generation call (prompt processing + each decoding step with KV cache)."""

from contextlib import contextmanager

from src.model_utils import get_module_by_path


@contextmanager
def inject_vector(model, layer_module_path, vector, alpha):
    """Context manager that registers a forward hook adding alpha*vector to the
    output of the module at layer_module_path, for the lifetime of the `with` block.
    Safe to use with alpha=0 / vector=None as a no-op (still registers a hook that
    adds nothing, for symmetry with control runs if desired).
    """
    module = get_module_by_path(model, layer_module_path)

    def hook(mod, inp, out):
        if vector is None or alpha == 0:
            return out
        if isinstance(out, tuple):
            hidden = out[0]
            hidden = hidden + alpha * vector.to(hidden.dtype).to(hidden.device)
            return (hidden,) + out[1:]
        else:
            return out + alpha * vector.to(out.dtype).to(out.device)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_with_injection(
    model,
    tokenizer,
    layer_module_path,
    prompt_text,
    vector=None,
    alpha=0.0,
    max_new_tokens=60,
    device="cuda",
    skip_special_tokens=True,
):
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with inject_vector(model, layer_module_path, vector, alpha):
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False for thinking-mode runs: Qwen3's
    # </think> marker must survive decoding so the reasoning/final-answer
    # split downstream (src/thinking_split.py) can find it.
    text = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
    return text.strip(), gen_ids.shape[0]
