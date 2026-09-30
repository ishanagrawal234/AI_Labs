"""
Probability Normalisation Test
================================
Tests that for every current word w, the sum of P(v | w) over all v equals 1.
This verifies the correctness of the conditional probability tables.
"""

from first_order_model import (
    tokenise, count_transitions, build_cpt, training_sentences
)
from second_order_model import (
    count_trigram_transitions, build_trigram_cpt
)


def test_bigram_normalisation(probabilities):
    """Test that sum_v P(v | w) = 1 for every word w (first-order model)."""
    print("=== First-Order (Bigram) Normalisation Test ===")
    all_pass = True
    for word in sorted(probabilities):
        total = sum(probabilities[word].values())
        status = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {word:12s}  sum = {total:.6f}  [{status}]")
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_trigram_normalisation(probabilities):
    """Test that sum_v P(v | w1, w2) = 1 for every context (w1, w2) (second-order model)."""
    print("\n=== Second-Order (Trigram) Normalisation Test ===")
    all_pass = True
    for context in sorted(probabilities, key=str):
        total = sum(probabilities[context].values())
        status = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {str(context):35s}  sum = {total:.6f}  [{status}]")
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    tokenised = tokenise(training_sentences)

    # First-order
    bigram_counts = count_transitions(tokenised)
    bigram_probs = build_cpt(bigram_counts)
    test_bigram_normalisation(bigram_probs)

    # Second-order
    trigram_counts = count_trigram_transitions(tokenised)
    trigram_probs = build_trigram_cpt(trigram_counts)
    test_trigram_normalisation(trigram_probs)
