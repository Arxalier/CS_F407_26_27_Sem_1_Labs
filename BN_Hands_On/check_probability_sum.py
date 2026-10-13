import math
from collections import defaultdict, Counter

# Training corpus from lab
D = """
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
"""

# Prepare training data with <START> and <END> tokens
raw_sentences = [line.strip().split() for line in D.strip().split('\n') if line.strip()]
training_data = [["<START>"] + sentence + ["<END>"] for sentence in raw_sentences]

# 1. Count transitions between consecutive tokens
transitions = defaultdict(Counter)
for sentence in training_data:
    for i in range(len(sentence) - 1):
        transitions[sentence[i]][sentence[i + 1]] += 1

# 2. Compute conditional distribution P(Xt | Xt-1)
probabilities = defaultdict(dict)
for current_token, next_tokens in transitions.items():
    total_transitions = sum(next_tokens.values())
    for next_token, count in next_tokens.items():
        probabilities[current_token][next_token] = count / total_transitions


def verify_probability_distributions():
    all_valid = True
    print(f"{'Word':<10} | {'Sum P(v|w)':<12} | Status")
    print("-" * 35)

    for word in probabilities:
        total = sum(probabilities[word].values())
        
        # In floating-point arithmetic, check if total is approximately 1.0
        is_valid = math.isclose(total, 1.0, rel_tol=1e-9)
        
        print(f"{repr(word):<10} | {total:<12.6f} | {'✓ PASS' if is_valid else '✗ FAIL'}")
        
        if not is_valid:
            all_valid = False

    print("-" * 35)
    print(f"All conditional distributions valid: {all_valid}")
    return all_valid

# Run the test
verify_probability_distributions()
