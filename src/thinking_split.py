"""Splits a reasoning model's raw generation into its thinking block and
final answer. Used by the v6 experiment runner and the removal-test /
entropy-trace Modal functions (scripts/modal_app.py) to separate a
model's chain of thought from its final answer before scoring.

Different model families spell their reasoning-block delimiters
differently. Qwen3/DeepSeek-R1-distills use <think>...</think>; Gemma 4
uses a "<|channel>thought\\n...<channel|>" convention instead. With
enable_thinking=True, the chat template appends the opening tag to the
PROMPT (not the generated text), so the model's generated text is
"{reasoning}{close_tag}{final answer}". We decode with
skip_special_tokens=False so the close marker survives even if the
tokenizer treats it as a special token, then split on its first
occurrence. If no close marker appears, the model never finished
thinking within the token budget -- the whole generation is reasoning
and there is no final answer to score.
"""

THINKING_MARKERS = [
    ("<think>", "</think>"),
    ("<|channel>thought", "<channel|>"),
]


def split_thinking_and_final(raw_text, markers=None):
    for open_tag, close_tag in (markers or THINKING_MARKERS):
        if close_tag not in raw_text:
            continue
        text = raw_text
        if open_tag in text:
            # defensive: only matters if the template didn't pre-seed the
            # opening tag into the prompt for some model/config variant.
            text = text.split(open_tag, 1)[1]
        thinking_text, final_text = text.split(close_tag, 1)
        return thinking_text.strip(), final_text.strip(), True

    # no known close marker found -- ran out of budget before finishing,
    # or never entered a reasoning block at all.
    text = raw_text
    for open_tag, _ in (markers or THINKING_MARKERS):
        if open_tag in text:
            text = text.split(open_tag, 1)[1]
            break
    return text.strip(), "", False
