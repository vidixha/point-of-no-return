# Point of No Return: Does Concept Injection Break Reasoning Models?

We test whether activation steering (concept injection) that is safe for
short, direct-answer generation still breaks extended chain-of-thought
generation, and whether removing the injected direction partway through
generation lets a reasoning model recover.

Main result: a fixed injection strength, validated as safe on short
answers, drives valid completion to 0% in 105 of 112
model-direction-depth combinations across seven open-weight reasoning
models. Removing the injected direction after enough tokens does not
restore a valid completion, in every model with a valid baseline: past a
model-dependent point, the model's trajectory is already fixed.

Full writeup: [`TRACK3_REPORT.pdf`](TRACK3_REPORT.pdf).

## Models and GPUs

All runs used serverless GPU infrastructure. Models with 12B+ parameters
need more headroom than a 24GB card comfortably gives alongside an
800-token generation buffer, so they ran on a larger card.

| Model | Architecture | Params | GPU |
|---|---|---|---|
| Qwen3-1.7B | Qwen3 | 1.7B | L4 (24GB) |
| Qwen3-4B | Qwen3 | 4B | L4 (24GB) |
| Qwen3-8B | Qwen3 | 8B | L4 (24GB) |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2 | 7B | L4 (24GB) |
| DeepSeek-R1-Distill-Llama-8B | Llama | 8B | L4 (24GB) |
| Phi-4-mini-reasoning | Phi-3 | ~4B | L4 (24GB) |
| Gemma 4 12B | Gemma 4 | 12B | A100 (40GB) |

## Repo layout

```
src/         intervention, model-loading, parsing, and orchestration code
scripts/     analysis, rescoring, and figure-generation scripts
TRACK3_REPORT.pdf   full paper
```

GPU orchestration code and raw per-run logs are not included in this repo.

## Method summary

**Interventions**
1. Single-vector: `v = activation(concept_prompt) - activation(neutral_prompt)`
   at one layer/position, unit-normalized.
2. Multi-example contrastive: 8 paraphrased concept/neutral prompt pairs per
   concept, each diffed and unit-normalized, then averaged and renormalized.

During elicitation, `alpha * v` is added to the residual stream at every
token position at the chosen layer, via a forward hook on the decoder
block. `alpha` is a multiplier on the measured mean residual-stream norm
at the injection layer, not an arbitrary raw scale. See `src/injection.py`
and `src/generality_experiment.py`.

**The removal test.** Inject the concept vector for only the first `k`
generated tokens, then remove it and let the model continue unsteered for
the rest of the generation budget. Sweeping `k` finds the point past which
recovery never happens.

**Scoring** (`src/parsing.py`, `src/thinking_split.py`): transparent
keyword and repetition-loop rules, not an LLM judge, so results are
reproducible without a second model in the loop.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

```bash
python3 scripts/test_parsing.py           # response-scoring unit tests
python3 scripts/test_thinking_split.py    # thinking/final-answer split unit tests
python3 scripts/test_coherence.py         # degenerate-output detection unit tests
```

## Analyze existing results

```bash
python3 scripts/analyze_results.py results/<results_file>.csv
```

If a scoring bug is found after the fact, fix `src/parsing.py`, add a
regression test, then re-score saved raw responses without any new GPU
calls:

```bash
python3 scripts/rescore_coherence.py results/<results_file>.csv
```

## Reproducibility

- Model revisions are pinned to exact Hugging Face commit SHAs.
- Generation is greedy (`do_sample=False`), so outputs are deterministic
  given the same weights and revision.
- All raw model responses are saved unfiltered, with no manual selection
  of "successful" examples.
