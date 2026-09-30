"""
First-Order Autoregressive Language Model
==========================================
A simple Bayesian-network-based language model that estimates P(Xt | Xt-1)
from transition counts. Supports greedy and sampling-based generation.
"""

import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
training_sentences = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START = "<START>"
END = "<END>"


def tokenise(sentences):
    """Convert raw sentences into lists of tokens with START/END markers."""
    tokenised = []
    for s in sentences:
        tokens = [START] + s.strip().lower().split() + [END]
        tokenised.append(tokens)
    return tokenised


# ---------------------------------------------------------------------------
# Count transitions  (bigrams)
# ---------------------------------------------------------------------------
def count_transitions(tokenised_sentences):
    """
    Count how many times each word wj follows each word wi.
    Returns a dict-of-dicts: counts[wi][wj] = C(wi, wj).
    """
    counts = defaultdict(lambda: defaultdict(int))
    for tokens in tokenised_sentences:
        for i in range(len(tokens) - 1):
            counts[tokens[i]][tokens[i + 1]] += 1
    return counts


# ---------------------------------------------------------------------------
# Build conditional probability table  P(Xt | Xt-1)
# ---------------------------------------------------------------------------
def build_cpt(counts):
    """
    Convert raw counts into conditional probabilities.
    probabilities[wi][wj] = P(wj | wi) = C(wi,wj) / sum_k C(wi,wk)
    """
    probabilities = {}
    for wi, followers in counts.items():
        total = sum(followers.values())
        probabilities[wi] = {wj: c / total for wj, c in followers.items()}
    return probabilities


# ---------------------------------------------------------------------------
# Display the CPT for a given word
# ---------------------------------------------------------------------------
def display_distribution(probabilities, word):
    """Print the next-word distribution for a given preceding word."""
    if word not in probabilities:
        print(f"  No transitions observed for '{word}'.")
        return
    print(f"  P(Xt+1 | Xt = '{word}'):")
    for next_word, prob in sorted(probabilities[word].items(),
                                   key=lambda x: -x[1]):
        print(f"    P({next_word} | {word}) = {prob:.4f}")


# ---------------------------------------------------------------------------
# Predict most probable next word  (argmax)
# ---------------------------------------------------------------------------
def predict_most_probable(probabilities, word):
    """Return argmax_w P(w | word)."""
    if word not in probabilities:
        return None
    return max(probabilities[word], key=probabilities[word].get)


# ---------------------------------------------------------------------------
# Sample next word from the distribution
# ---------------------------------------------------------------------------
def sample_next_word(probabilities, word):
    """Sample from P(w | word) using the distribution."""
    if word not in probabilities:
        return None
    words = list(probabilities[word].keys())
    probs = list(probabilities[word].values())
    return random.choices(words, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Generate a sentence
# ---------------------------------------------------------------------------
def generate_sentence(probabilities, mode="sampling", max_length=50):
    """
    Generate a sentence.
    mode = "sampling"  -> sample from P(w | w_prev)   [Mode B]
    mode = "greedy"    -> always pick argmax P(w | w_prev)  [Mode A]
    """
    sentence = []
    current = START
    for _ in range(max_length):
        if mode == "greedy":
            next_word = predict_most_probable(probabilities, current)
        else:
            next_word = sample_next_word(probabilities, current)

        if next_word is None or next_word == END:
            break
        sentence.append(next_word)
        current = next_word
    return " ".join(sentence)


# ---------------------------------------------------------------------------
# Test probability normalisation
# ---------------------------------------------------------------------------
def test_normalisation(probabilities):
    """For every word w, check that sum_v P(v | w) ≈ 1."""
    print("\n=== Probability Normalisation Test ===")
    all_pass = True
    for word in sorted(probabilities):
        total = sum(probabilities[word].values())
        status = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {word:12s}  sum = {total:.6f}  [{status}]")
    if all_pass:
        print("  All distributions normalise to 1.0. [PASS]")
    else:
        print("  WARNING: Some distributions do NOT normalise to 1.0!")
    return all_pass


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    # 1. Tokenise
    tokenised = tokenise(training_sentences)
    print("=== Tokenised Sentences ===")
    for t in tokenised:
        print("  ", " ".join(t))

    # 2. Count transitions
    counts = count_transitions(tokenised)

    # 3. Build CPT
    probabilities = build_cpt(counts)

    # 4. Display distributions for selected words
    print("\n=== Conditional Probability Distributions ===")
    for word in ["<START>", "the", "cat", "dog", "sat", "ran", "on", "to"]:
        display_distribution(probabilities, word)
        best = predict_most_probable(probabilities, word)
        if best:
            print(f"    -> argmax: '{best}'")
        print()

    # 5. Test normalisation
    test_normalisation(probabilities)

    # 6. Generate 20 sentences (sampling)
    print("\n=== Generated Sentences (Sampling – Mode B) ===")
    sampling_sentences = []
    for i in range(20):
        s = generate_sentence(probabilities, mode="sampling")
        sampling_sentences.append(s)
        print(f"  {i+1:2d}. {s}")

    # 7. Generate 5 sentences (greedy – Mode A)
    print("\n=== Generated Sentences (Greedy – Mode A) ===")
    greedy_sentences = []
    for i in range(5):
        s = generate_sentence(probabilities, mode="greedy")
        greedy_sentences.append(s)
        print(f"  {i+1}. {s}")

    # 8. Generate 5 sentences (sampling – Mode B)
    print("\n=== Generated Sentences (Sampling – Mode B) ===")
    sampling_five = []
    for i in range(5):
        s = generate_sentence(probabilities, mode="sampling")
        sampling_five.append(s)
        print(f"  {i+1}. {s}")

    # 9. Save generated sentences to file
    with open("generated_sentences_first_order.txt", "w") as f:
        f.write("=== First-Order Model: Sampling (20 sentences) ===\n")
        for i, s in enumerate(sampling_sentences, 1):
            f.write(f"{i:2d}. {s}\n")
        f.write("\n=== First-Order Model: Greedy (5 sentences) ===\n")
        for i, s in enumerate(greedy_sentences, 1):
            f.write(f"{i}. {s}\n")
        f.write("\n=== First-Order Model: Sampling (5 sentences) ===\n")
        for i, s in enumerate(sampling_five, 1):
            f.write(f"{i}. {s}\n")
    print("\nGenerated sentences saved to 'generated_sentences_first_order.txt'.")
