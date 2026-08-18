# Point of No Return: Does Concept Injection Break Reasoning Models?

We test whether activation steering (concept injection) that is safe for
short, direct-answer generation still breaks extended chain-of-thought
generation, and whether removing the injected direction partway through
generation lets a reasoning model recover.

TL;DR: A fixed injection strength that is safe for short answers reduces valid completion to 0% in 105 of 112 model-direction-depth combinations across seven open-weight reasoning models. Removing the injected direction after enough tokens does not restore valid completion. For every model with a valid baseline, there is a model-dependent point after which the model's trajectory is already fixed (which we call the point-of-no-return). 

Full writeup: [`TRACK3_REPORT.pdf`](TRACK3_REPORT.pdf).

## Models and GPUs


| Model | Architecture | Params | GPU |
|---|---|---|---|
| Qwen3-1.7B | Qwen3 | 1.7B | L4 (24GB) |
| Qwen3-4B | Qwen3 | 4B | L4 (24GB) |
| Qwen3-8B | Qwen3 | 8B | L4 (24GB) |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2 | 7B | L4 (24GB) |
| DeepSeek-R1-Distill-Llama-8B | Llama | 8B | L4 (24GB) |
| Phi-4-mini-reasoning | Phi-3 | ~4B | L4 (24GB) |
| Gemma 4 12B | Gemma 4 | 12B | A100 (40GB) |

## Repo structure

```
TRACK3_REPORT.pdf   full paper
README.md           this file
requirements.txt    Python dependencies
src/                intervention, model-loading, parsing, and orchestration code
scripts/            analysis, rescoring, figure-generation, and test scripts
```



| File | What it does |
|---|---|
| `injection.py` | The intervention- adds `alpha * v` to the residual stream at a chosen layer, at every token position, for every forward pass during generation. |
| `vectors.py` | Builds concept-injection directions: a single contrastive-pair vector, and the multi-example contrastive vector (8 paraphrase pairs, averaged and renormalized) used throughout the paper. |
| `model_utils.py` | Model/tokenizer loading and residual-stream module lookup (handles both standard and multimodal/"unified" architectures like Gemma 4). |
| `parsing.py` | Scores a free-text response for concept detection/identification using transparent multilingual keyword rules (no LLM judge). |
| `thinking_split.py` | Splits a reasoning model's raw output into its thinking block and final answer, handling both `<think>...</think>` and Gemma 4's `<\|channel>thought...<channel\|>` marker conventions. |
| `coherence.py` | Flags degenerate/repetitive output (the "I'm a student. I'm a student..." failure mode) with a cheap n-gram heuristic, not a model judge. |
| `run_experiment.py` | Shared utility: measures the mean residual-stream activation norm at a layer, used to set `alpha` in physically meaningful units. |
| `generality_experiment.py` | The main experiment loop- runs the no-injection / concept-vector / random-vector conditions across templates for one model and writes result rows. |
| `schema.py` | The CSV row schema and incremental `ResultWriter` used by the experiment loop. |
| `config.py` | Loads the YAML concept/language/template configs (not included in this repo; see note below). |

**`scripts/`** — analysis, rescoring, figure generation, and tests:

| File | What it does |
|---|---|
| `analyze_results.py` | Computes the corrected `genuinely_valid` completion metric (accounts for the Gemma marker-closes-while-degenerate bug) across all seven models' result CSVs, producing the paper's headline numbers. |
| `rescore_coherence.py` | Re-scores `thinking_is_degenerate`/`final_is_degenerate` in already-saved result CSVs using a corrected `coherence.py`|
| `make_critical_window_figure.py` | Generates the removal-ablation figure (Figure 2 / `critical_window.png`) directly from the saved removal-test JSON files. |
| `make_track3_figures.py` | Generates the two main-results summary figures used in the report, computed from the saved result CSVs. |
| `test_parsing.py` | Unit tests for `parsing.py`'s scoring rules against hand-written EN/ES/ZH example responses. |
| `test_thinking_split.py` | Unit tests for the thinking/final-answer split, including both marker conventions and the truncated-budget case. |
| `test_coherence.py` | Regression tests for `coherence.py` |



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


