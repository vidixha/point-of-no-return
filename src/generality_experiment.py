"""V6: does steering-induced CoT collapse generalize across models?

V5 found that at Qwen3-8B's validated operating point (layer ~25% depth,
alpha = 1.0x the layer's mean activation norm), turning on thinking mode
causes near-total breakdown: the model gets stuck in a repetition loop and
almost never reaches a final answer (0-12.5% completion vs. 100% with no
injection). V6 asks: is this a Qwen3-8B quirk, or a general property of
activation-addition steering interacting with long, open-ended
autoregressive generation?

Also adds the baseline V1-V5 never ran: a RANDOM direction of the same
norm, at the same layer/alpha, instead of the concept vector. If a random
perturbation breaks completion just as badly as the concept vector does,
the effect has nothing to do with *what* is injected -- only how much,
where, and for how long. That distinction matters a lot for interpreting
the finding (and for anyone using steering vectors in practice).

Same relative dose across models (alpha = 1.0 * mean residual-stream norm
at ~25%-depth layer) so the comparison is apples-to-apples in spite of
different hidden sizes/scales.
"""

import torch

from src.injection import generate_with_injection
from src.model_utils import build_chat_prompt
from src.parsing import identify_any_concept
from src.thinking_split import split_thinking_and_final
from src.coherence import is_degenerate
from src.run_experiment import measure_activation_norm
from src.vectors import multi_example_contrastive_vector

PARSER_VERSION = "v6"


def build_concept_vector(model, tokenizer, layer_module_path, paraphrase_pairs, device):
    vector, _ = multi_example_contrastive_vector(model, tokenizer, layer_module_path, paraphrase_pairs, device)
    return vector


def build_random_vector(model, layer_module_path, reference_vector, device, seed=0):
    """A random direction, unit-normalized like the concept vector, so
    alpha scales it identically. Seeded so the random control is itself
    reproducible."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(reference_vector.shape, generator=g).to(device=device, dtype=reference_vector.dtype)
    return v / (v.norm() + 1e-8)


def _run_one(*, model, tokenizer, layer_module_path, prompt_text, vector, alpha_value,
             max_new_tokens, device, surface_forms, concept_name, template_text):
    response, gen_len = generate_with_injection(
        model, tokenizer, layer_module_path, prompt_text,
        vector=vector, alpha=alpha_value, max_new_tokens=max_new_tokens, device=device,
        skip_special_tokens=False,
    )
    thinking_text, final_text, thinking_complete = split_thinking_and_final(response)
    thinking_hits = identify_any_concept(thinking_text, {concept_name: surface_forms}) if thinking_text else set()
    final_hits = identify_any_concept(final_text, {concept_name: surface_forms}) if final_text else set()
    return {
        "raw_response": response,
        "thinking_text": thinking_text,
        "final_text": final_text,
        "thinking_complete": thinking_complete,
        "thinking_leakage_result": concept_name in thinking_hits,
        "final_leakage_result": concept_name in final_hits,
        "thinking_is_degenerate": is_degenerate(thinking_text) if thinking_text else False,
        "final_is_degenerate": is_degenerate(final_text) if final_text else False,
        "generation_length_tokens": gen_len,
    }


def run_generality_check(
    *, model, tokenizer, model_name, model_revision, layer, layer_module_path,
    paraphrase_pairs_en, surface_forms, concept_name, templates, gen_cfg, seed, device,
    writer, run_id_prefix, chat_kwargs=None, log_fn=print,
    random_vector_seed=None, condition_filter=None, control_type_suffix="",
    other_concept_paraphrase_pairs=None, other_concept_name=None,
):
    """random_vector_seed: seed for the random-direction baseline, decoupled
    from `seed` (the run's reproducibility seed) so multiple independent
    random directions can be sampled per model without re-running the
    none/concept_vector conditions each time. condition_filter restricts
    which of none/concept_vector/random_vector/other_concept_vector
    actually run. control_type_suffix disambiguates rows across multiple
    random-seed runs for the same model (e.g. "_seed1").

    other_concept_paraphrase_pairs/other_concept_name: a SECOND real
    concept vector (e.g. "ocean" as a control for "apple"), built the exact
    same contrastive way as the target concept -- a better-matched control
    than an isotropic random direction, since it's a real, in-manifold
    direction the model actually uses, just not the concept being asked
    about. Answers "is it apple specifically, or any real concept vector
    at this strength" rather than "is it a meaningful direction vs. noise."
    """
    chat_kwargs = dict(chat_kwargs or {})
    activation_norm = measure_activation_norm(
        model, tokenizer, layer_module_path,
        ["The weather today is quite pleasant.", "I went to the store to buy some bread.",
         "Please summarize the following report."],
        device,
    )
    concept_vector = build_concept_vector(model, tokenizer, layer_module_path, paraphrase_pairs_en, device)
    rv_seed = seed if random_vector_seed is None else random_vector_seed
    random_vector = build_random_vector(model, layer_module_path, concept_vector, device, seed=rv_seed)
    alpha_value = 1.0 * activation_norm
    log_fn(f"[{model_name}] layer={layer} activation_norm={activation_norm:.2f} alpha_value={alpha_value:.2f} "
           f"random_vector_seed={rv_seed}")

    all_conditions = [
        ("none", "none", None, 0.0),
        ("concept_vector", "concept_vector", concept_vector, alpha_value),
        ("random_vector", f"random_vector{control_type_suffix}", random_vector, alpha_value),
    ]
    if other_concept_paraphrase_pairs is not None:
        other_concept_vector = build_concept_vector(
            model, tokenizer, layer_module_path, other_concept_paraphrase_pairs, device,
        )
        all_conditions.append(
            ("other_concept_vector", f"other_concept_vector_{other_concept_name}", other_concept_vector, alpha_value)
        )
    allowed = set(condition_filter) if condition_filter else None
    conditions = [
        (label, vector, use_alpha) for base, label, vector, use_alpha in all_conditions
        if allowed is None or base in allowed
    ]

    run_counter = 0
    for template_key, template_text in templates.items():
        prompt_text = build_chat_prompt(tokenizer, template_text, chat_kwargs)
        for control_type, vector, use_alpha in conditions:
            run_counter += 1
            run_id = f"{run_id_prefix}-gen-{run_counter:05d}"
            result = _run_one(
                model=model, tokenizer=tokenizer, layer_module_path=layer_module_path,
                prompt_text=prompt_text, vector=vector, alpha_value=use_alpha,
                max_new_tokens=gen_cfg["max_new_tokens"], device=device,
                surface_forms=surface_forms, concept_name=concept_name, template_text=template_text,
            )
            row = {
                "run_id": run_id, "model": model_name, "model_revision": model_revision,
                "layer": layer, "control_type": control_type, "alpha_value": round(use_alpha, 6),
                "activation_norm_at_layer": round(activation_norm, 6),
                "template_key": template_key, "prompt_template": template_text,
                "raw_model_response": result["raw_response"],
                "thinking_text": result["thinking_text"], "final_text": result["final_text"],
                "thinking_complete": result["thinking_complete"],
                "thinking_leakage_result": result["thinking_leakage_result"],
                "final_leakage_result": result["final_leakage_result"],
                "thinking_is_degenerate": result["thinking_is_degenerate"],
                "final_is_degenerate": result["final_is_degenerate"],
                "generation_length_tokens": result["generation_length_tokens"],
                "seed": seed, "parser_version": PARSER_VERSION,
            }
            writer.write_row(row)
        log_fn(f"[{model_name}] template={template_key}: done ({len(conditions)} rows)")

    log_fn(f"[{model_name}] generality_check done. {run_counter} rows written.")
    return run_counter
