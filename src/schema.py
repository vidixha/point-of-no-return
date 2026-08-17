"""V6 result row schema: cross-model generality check for the steering-
induced CoT collapse found in V5, plus the random-vector baseline."""

import csv
import os

COLUMNS = [
    "run_id",
    "model",
    "model_revision",
    "layer",
    "control_type",              # none | concept_vector | random_vector
    "alpha_value",
    "activation_norm_at_layer",
    "template_key",
    "prompt_template",
    "raw_model_response",
    "thinking_text",
    "final_text",
    "thinking_complete",
    "thinking_leakage_result",
    "final_leakage_result",
    "thinking_is_degenerate",
    "final_is_degenerate",
    "generation_length_tokens",
    "seed",
    "parser_version",
]


class ResultWriter:
    def __init__(self, path):
        self.path = path
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write_row(self, row):
        clean = {k: row.get(k, "") for k in COLUMNS}
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self):
        self._fh.close()
