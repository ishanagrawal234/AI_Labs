"""
Second-Order Autoregressive Language Model
============================================
A Bayesian-network-based language model that estimates P(Xt | Xt-2, Xt-1)
from counts of observed triples. Supports greedy and sampling-based generation.
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
# Count transitions  (trigrams)
# ---------------------------------------------------------------------------
def count_trigram_transitions(tokenised_sentences):
    """
    Count how many times word wk follows the pair (wi, wj).
    Returns: counts[(wi, wj)][wk] = C(wi, wj, wk)
    """
    counts = defaultdict(lambda: defaultdict(int))
    for tokens in tokenised_sentences:
        for i in range(len(tokens) - 2):
            context = (tokens[i], tokens[i + 1])
            next_word = tokens[i + 2]
            counts[context][next_word] += 1
    return counts


# We also need first-order counts for the first transition P(X2 | X1)
def count_bigram_transitions(tokenised_sentences):
    """Count bigram transitions for fallback / initial step."""
    counts = defaultdict(lambda: defaultdict(int))
    for tokens in tokenised_sentences:
        for i in range(len(tokens) - 1):
            counts[tokens[i]][tokens[i + 1]] += 1
    return counts


# ---------------------------------------------------------------------------
# Build CPTs
# ---------------------------------------------------------------------------
def build_trigram_cpt(counts):
    """
    P(wk | wi, wj) = C(wi, wj, wk) / sum_m C(wi, wj, wm)
    """
    probabilities = {}
    for context, followers in counts.items():
        total = sum(followers.values())
        probabilities[context] = {wk: c / total for wk, c in followers.items()}
    return probabilities


def build_bigram_cpt(counts):
    """P(wj | wi) = C(wi, wj) / sum_k C(wi, wk)"""
    probabilities = {}
    for wi, followers in counts.items():
        total = sum(followers.values())
        probabilities[wi] = {wj: c / total for wj, c in followers.items()}
    return probabilities


# ---------------------------------------------------------------------------
# Display distributions
# ---------------------------------------------------------------------------
def display_trigram_distribution(probabilities, context):
    """Print P(Xt | Xt-2, Xt-1) for a given context pair."""
    if context not in probabilities:
        print(f"  No transitions observed for context {context}.")
        return
    print(f"  P(Xt | Xt-2='{context[0]}', Xt-1='{context[1]}'):")
    for next_word, prob in sorted(probabilities[context].items(),
                                   key=lambda x: -x[1]):
        print(f"    P({next_word} | {context[0]}, {context[1]}) = {prob:.4f}")


# ---------------------------------------------------------------------------
# Predict / Sample
# ---------------------------------------------------------------------------
def predict_most_probable_trigram(probabilities, context):
    if context not in probabilities:
        return None
    return max(probabilities[context], key=probabilities[context].get)


def sample_next_word_trigram(probabilities, context):
    if context not in probabilities:
        return None
    words = list(probabilities[context].keys())
    probs = list(probabilities[context].values())
    return random.choices(words, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Generate a sentence (second-order)
# ---------------------------------------------------------------------------
def generate_sentence(bigram_probs, trigram_probs,
                      mode="sampling", max_length=50):
    """
    Generate using second-order model.
    First word: sample from P(X2 | X1=<START>) using bigram.
    Subsequent words: sample from P(Xt | Xt-2, Xt-1) using trigram.
    Falls back to bigram if trigram context is unseen.
    """
    sentence = []

    # --- First word ---
    if mode == "greedy":
        first = max(bigram_probs[START], key=bigram_probs[START].get)
    else:
        words = list(bigram_probs[START].keys())
        probs = list(bigram_probs[START].values())
        first = random.choices(words, weights=probs, k=1)[0]

    if first == END:
        return ""
    sentence.append(first)
    prev2 = START
    prev1 = first

    # --- Subsequent words ---
    for _ in range(max_length):
        context = (prev2, prev1)
        if context in trigram_probs:
            if mode == "greedy":
                next_word = predict_most_probable_trigram(trigram_probs, context)
            else:
                next_word = sample_next_word_trigram(trigram_probs, context)
        elif prev1 in bigram_probs:
            # Fallback to bigram
            if mode == "greedy":
                next_word = max(bigram_probs[prev1],
                                key=bigram_probs[prev1].get)
            else:
                words = list(bigram_probs[prev1].keys())
                probs = list(bigram_probs[prev1].values())
                next_word = random.choices(words, weights=probs, k=1)[0]
        else:
            break

        if next_word is None or next_word == END:
            break
        sentence.append(next_word)
        prev2 = prev1
        prev1 = next_word

    return " ".join(sentence)


# ---------------------------------------------------------------------------
# Test probability normalisation
# ---------------------------------------------------------------------------
def test_normalisation(trigram_probs):
    print("\n=== Trigram Probability Normalisation Test ===")
    all_pass = True
    for context in sorted(trigram_probs, key=str):
        total = sum(trigram_probs[context].values())
        status = "OK" if abs(total - 1.0) < 1e-9 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {str(context):35s}  sum = {total:.6f}  [{status}]")
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
    trigram_counts = count_trigram_transitions(tokenised)
    bigram_counts = count_bigram_transitions(tokenised)

    # 3. Build CPTs
    trigram_probs = build_trigram_cpt(trigram_counts)
    bigram_probs = build_bigram_cpt(bigram_counts)

    # 4. Display distributions for selected contexts
    print("\n=== Trigram Conditional Probability Distributions ===")
    sample_contexts = [
        ("<START>", "the"),
        ("the", "cat"),
        ("the", "dog"),
        ("cat", "sat"),
        ("cat", "ran"),
        ("dog", "sat"),
        ("dog", "ran"),
        ("sat", "on"),
        ("on", "the"),
        ("ran", "to"),
        ("to", "the"),
        ("the", "mat"),
        ("the", "rug"),
        ("the", "park"),
    ]
    for ctx in sample_contexts:
        display_trigram_distribution(trigram_probs, ctx)
        best = predict_most_probable_trigram(trigram_probs, ctx)
        if best:
            print(f"    -> argmax: '{best}'")
        print()

    # 5. Test normalisation
    test_normalisation(trigram_probs)

    # 6. Count parameters
    num_trigram_params = sum(len(v) for v in trigram_probs.values())
    num_bigram_params = sum(len(v) for v in bigram_probs.values())
    print(f"\n=== Model Statistics ===")
    print(f"  Trigram model: {len(trigram_probs)} contexts, "
          f"{num_trigram_params} distinct parameters")
    print(f"  Bigram model:  {len(bigram_probs)} contexts, "
          f"{num_bigram_params} distinct parameters")

    # 7. Count zero-probability contexts
    all_words = set()
    for tokens in tokenised:
        all_words.update(tokens)
    possible_bigram_contexts = len(all_words)
    possible_trigram_contexts = len(all_words) ** 2
    observed_trigram_contexts = len(trigram_probs)
    zero_trigram = possible_trigram_contexts - observed_trigram_contexts
    print(f"  Possible trigram contexts (V^2): {possible_trigram_contexts}")
    print(f"  Observed trigram contexts:       {observed_trigram_contexts}")
    print(f"  Zero-probability contexts:       {zero_trigram}")

    # 8. Generate 20 sentences (sampling)
    print("\n=== Generated Sentences (Second-Order, Sampling) ===")
    sampling_sentences = []
    for i in range(20):
        s = generate_sentence(bigram_probs, trigram_probs, mode="sampling")
        sampling_sentences.append(s)
        print(f"  {i+1:2d}. {s}")

    # 9. Generate 5 greedy + 5 sampling
    print("\n=== Greedy (Mode A) ===")
    greedy_sentences = []
    for i in range(5):
        s = generate_sentence(bigram_probs, trigram_probs, mode="greedy")
        greedy_sentences.append(s)
        print(f"  {i+1}. {s}")

    print("\n=== Sampling (Mode B) ===")
    sampling_five = []
    for i in range(5):
        s = generate_sentence(bigram_probs, trigram_probs, mode="sampling")
        sampling_five.append(s)
        print(f"  {i+1}. {s}")

    # 10. Save generated sentences
    with open("generated_sentences_second_order.txt", "w") as f:
        f.write("=== Second-Order Model: Sampling (20 sentences) ===\n")
        for i, s in enumerate(sampling_sentences, 1):
            f.write(f"{i:2d}. {s}\n")
        f.write("\n=== Second-Order Model: Greedy (5 sentences) ===\n")
        for i, s in enumerate(greedy_sentences, 1):
            f.write(f"{i}. {s}\n")
        f.write("\n=== Second-Order Model: Sampling (5 sentences) ===\n")
        for i, s in enumerate(sampling_five, 1):
            f.write(f"{i}. {s}\n")
    print("\nGenerated sentences saved to 'generated_sentences_second_order.txt'.")
