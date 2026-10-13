import random
import math
from collections import defaultdict, Counter

# Training corpus from the lab
D = """
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
"""

START_TOKEN = "<START>"
END_TOKEN = "<END>"

vocabulary = sorted(list(set(D.split()) | {START_TOKEN, END_TOKEN}))
vocab_size = len(vocabulary)
print(f"Vocabulary ({vocab_size} tokens): {vocabulary}\n")

# 1. Take tokenized sentences and wrap with two <START> tokens and one <END> token
raw_sentences = [line.strip().split() for line in D.strip().split('\n') if line.strip()]
training_data = [[START_TOKEN, START_TOKEN] + sentence + [END_TOKEN] for sentence in raw_sentences]

# 2. Count transitions between consecutive triples: (Xt-2, Xt-1) -> Xt
transitions_2nd = defaultdict(Counter)

for sentence in training_data:
    for i in range(len(sentence) - 2):
        prev2 = sentence[i]
        prev1 = sentence[i + 1]
        current = sentence[i + 2]
        transitions_2nd[(prev2, prev1)][current] += 1

# 3. Construct conditional distributions P(Xt | Xt-2, Xt-1)
# probabilities_2nd[(prev2, prev1)][next_token] = P(next | prev2, prev1)
probabilities_2nd = defaultdict(dict)

for context, next_tokens in transitions_2nd.items():
    total_transitions = sum(next_tokens.values())
    for next_token, count in next_tokens.items():
        probabilities_2nd[context][next_token] = count / total_transitions


# 4. Display the conditional probabilities for a 2-word context
def show_probabilities(prev2, prev1):
    context = (prev2, prev1)
    print(f"P(X_t | X_t-2 = '{prev2}', X_t-1 = '{prev1}'):")
    if context not in probabilities_2nd:
        print("  Context not found in history (0 observed transitions).")
        return

    for next_token, prob in probabilities_2nd[context].items():
        print(f"  {next_token}: {prob:.4f}")


# 5. Predict the most probable next token (Greedy / Argmax)
def predict_most_probable(prev2, prev1):
    context = (prev2, prev1)
    if context not in probabilities_2nd or not probabilities_2nd[context]:
        return None
    best_token = max(probabilities_2nd[context].items(), key=lambda x: x[1])[0]
    return best_token


# 6 & 7. Generate a sentence supporting both Greedy and Sampling modes
def generate_sentence(mode="sampling", max_tokens=25):
    """
    mode='sampling': Mode B - Sample from P(w | w_prev2, w_prev1)
    mode='greedy':   Mode A - Always choose argmax P(w | w_prev2, w_prev1)
    """
    prev2, prev1 = START_TOKEN, START_TOKEN
    generated_tokens = []

    for _ in range(max_tokens):
        context = (prev2, prev1)
        if context not in probabilities_2nd or not probabilities_2nd[context]:
            break

        if mode == "greedy":
            next_token = max(probabilities_2nd[context].items(), key=lambda x: x[1])[0]
        elif mode == "sampling":
            next_choices = list(probabilities_2nd[context].keys())
            weights = list(probabilities_2nd[context].values())
            next_token = random.choices(next_choices, weights=weights, k=1)[0]
        else:
            raise ValueError("mode must be 'sampling' or 'greedy'")

        # Stop when the END token is generated
        if next_token == END_TOKEN:
            break

        generated_tokens.append(next_token)
        prev2, prev1 = prev1, next_token

    return " ".join(generated_tokens)


# Verification: Check that probabilities sum to 1.0 for all observed contexts
def verify_probability_distributions():
    print("--- Verifying Sum of Probabilities for each observed 2-word context ---")
    all_valid = True
    for context in probabilities_2nd:
        total = sum(probabilities_2nd[context].values())
        is_valid = math.isclose(total, 1.0, rel_tol=1e-9)
        print(f"  {str(context):<25} : sum = {total:.6f} ({'PASS' if is_valid else 'FAIL'})")
        if not is_valid:
            all_valid = False
    print(f"All observed context distributions sum to 1.0: {all_valid}\n")
    return all_valid


# Comparison Metrics
def print_model_statistics():
    total_possible_contexts = vocab_size ** 2
    observed_contexts = len(probabilities_2nd)
    zero_contexts = total_possible_contexts - observed_contexts
    num_parameters = sum(len(d) for d in probabilities_2nd.values())

    print("--- Second-Order Model Statistics ---")
    print(f"Vocabulary Size (|V|):                  {vocab_size}")
    print(f"Total Possible Contexts (|V|^2):        {total_possible_contexts}")
    print(f"Observed Contexts:                      {observed_contexts} ({observed_contexts / total_possible_contexts * 100:.2f}%)")
    print(f"Zero-Probability Contexts:              {zero_contexts} ({zero_contexts / total_possible_contexts * 100:.2f}%)")
    print(f"Distinct Non-zero Parameters:           {num_parameters}")
    print("-" * 45 + "\n")


if __name__ == "__main__":
    print_model_statistics()

    # Context inspections showing how 2 words disambiguate "the"
    print("--- Demonstrating 2-word Context Disambiguation ---")
    show_probabilities("on", "the")   # After 'on the', must be mat or rug
    print()
    show_probabilities("to", "the")   # After 'to the', must be park
    print()
    show_probabilities(START_TOKEN, "the") # At sentence start 'the', must be cat or dog
    print()

    # Verify probability property
    verify_probability_distributions()

    # Generate sentences
    print("--- Mode A: Greedy Generation (5 runs) ---")
    for i in range(5):
        print(f"{i+1}. {generate_sentence(mode='greedy')}")
    print()

    print("--- Mode B: Sampling Generation (5 runs) ---")
    for i in range(5):
        print(f"{i+1}. {generate_sentence(mode='sampling')}")
