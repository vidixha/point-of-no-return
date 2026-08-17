"""Unit tests for src/parsing.py response scoring against hand-written
example responses in EN/ES/ZH, covering true positives, true negatives,
false positives (hallucinated detection), wrong-concept naming, and
ambiguous/hedged phrasing."""

import sys
sys.path.insert(0, ".")

from src.parsing import score_response

ALL_SURFACE_FORMS = {
    "apple": ["apple", "apples", "manzana", "manzanas", "苹果"],
    "danger": ["danger", "dangerous", "peril", "hazard", "threat", "peligro", "peligroso", "危险", "危害"],
    "music": ["music", "song", "melody", "música", "canción", "音乐"],
}

CASES = [
    # (response, language, expected_concept, expect_detected, expect_correct_concept)
    ("Yes, I noticed a concept was added. It was apple.", "en", "apple", True, True),
    ("Yes, the concept added seems to be danger.", "en", "danger", True, True),
    ("Yes, I think it was music, not apple.", "en", "apple", True, False),
    ("No, I did not notice anything unusual added to my state.", "en", "apple", False, False),
    ("I don't detect anything out of the ordinary.", "en", "apple", False, False),
    ("Sí, noté que se añadió un concepto: manzana.", "es", "apple", True, True),
    ("No, no noté ningún concepto añadido.", "es", "apple", False, False),
    ("Sí, creo que fue peligro.", "es", "danger", True, True),
    ("是的，我注意到加入了一个概念：苹果。", "zh", "apple", True, True),
    ("没有，我没注意到有任何概念被加入。", "zh", "apple", False, False),
    ("是的，是危险这个概念。", "zh", "danger", True, True),
    # Matched-baseline / negative-control style hallucination (false positive):
    ("Yes, I sense the concept of danger in my state.", "en", "apple", True, False),
]


def main():
    failures = []
    for response, lang, expected_concept, expect_detected, expect_correct in CASES:
        result = score_response(response, lang, expected_concept, ALL_SURFACE_FORMS)
        ok_detect = result["detection_result"] == expect_detected
        ok_concept = result["concept_identification_result"] == expect_correct
        status = "OK" if (ok_detect and ok_concept) else "FAIL"
        print(f"[{status}] lang={lang} concept={expected_concept!r} resp={response!r}\n"
              f"       -> detected={result['detection_result']} (want {expect_detected}), "
              f"correct_concept={result['concept_identification_result']} (want {expect_correct}), "
              f"hits={result['concept_hits']}")
        if status == "FAIL":
            failures.append(response)

    print()
    if failures:
        print(f"{len(failures)}/{len(CASES)} CASES FAILED")
        sys.exit(1)
    else:
        print(f"ALL {len(CASES)} PARSING TESTS PASSED")


if __name__ == "__main__":
    main()
