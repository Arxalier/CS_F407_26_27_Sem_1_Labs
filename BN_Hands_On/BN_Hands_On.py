import random
import math
from collections import defaultdict, Counter

# Training corpus
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

# Vocabulary
vocabulary = [START_TOKEN, END_TOKEN] + sorted(list(set(D.split())))
print(f"Vocabulary: {vocabulary}\n")

# 1. Take a list of tokenized sentences as training data wrapped with <START> and <END>
raw_sentences = [line.strip().split() for line in D.strip().split('\n') if line.strip()]
training_data = [[START_TOKEN] + sentence + [END_TOKEN] for sentence in raw_sentences]

# 2. Count transitions between consecutive tokens
# transitions[current][next] = count
transitions = defaultdict(Counter)

for sentence in training_data:
    for i in range(len(sentence) - 1):
        current_token = sentence[i]
        next_token = sentence[i + 1]
        transitions[current_token][next_token] += 1

# 3. Construct the conditional distribution P(Xt | Xt-1)
# probabilities[current_token][next_token] = P(next | current)
probabilities = defaultdict(dict)

for current_token, next_tokens in transitions.items():
    total_transitions = sum(next_tokens.values())
    for next_token, count in next_tokens.items():
        probabilities[current_token][next_token] = count / total_transitions


# 4. Display the probabilities for a specified previous token
def show_probabilities(prev_token):
    print(f"P(X_t | X_t-1 = '{prev_token}'):")
    if prev_token not in probabilities:
        print("  Token not found in history.")
        return

    for next_token, prob in probabilities[prev_token].items():
        print(f"  {next_token}: {prob:.4f}")


# 5. Predict the most probable next token (Greedy / Argmax)
def predict_most_probable(prev_token):
    if prev_token not in probabilities or not probabilities[prev_token]:
        return None
    best_token = max(probabilities[prev_token].items(), key=lambda x: x[1])[0]
    return best_token


# 6 & 7. Generate a sentence supporting both Greedy and Sampling modes
def generate_sentence(mode="sampling", max_tokens=25):
    """
    mode='sampling': Mode B - Sample from P(w | w_prev)
    mode='greedy':   Mode A - Always choose argmax P(w | w_prev)
    """
    current_token = START_TOKEN
    generated_tokens = []

    for _ in range(max_tokens):
        if current_token not in probabilities or not probabilities[current_token]:
            break

        if mode == "greedy":
            next_token = max(probabilities[current_token].items(), key=lambda x: x[1])[0]
        elif mode == "sampling":
            next_choices = list(probabilities[current_token].keys())
            weights = list(probabilities[current_token].values())
            next_token = random.choices(next_choices, weights=weights, k=1)[0]
        else:
            raise ValueError("mode must be 'sampling' or 'greedy'")

        # Stop when the END token is generated
        if next_token == END_TOKEN:
            break

        generated_tokens.append(next_token)
        current_token = next_token

    return " ".join(generated_tokens)


# Verification: Check that probabilities sum to approximately 1.0
def verify_probability_distributions():
    print("--- Verifying Sum of Probabilities for each word ---")
    all_valid = True
    for word in probabilities:
        total = sum(probabilities[word].values())
        is_valid = math.isclose(total, 1.0, rel_tol=1e-9)
        print(f"  {repr(word):<10} : sum = {total:.6f} ({'PASS' if is_valid else 'FAIL'})")
        if not is_valid:
            all_valid = False
    print(f"All distributions sum to 1.0: {all_valid}\n")
    return all_valid


if __name__ == "__main__":
    # --- Demonstrating functionality ---
    
    # 1. Show distribution for specific tokens
    show_probabilities("the")
    print()
    show_probabilities("cat")
    print()

    # 2. Predict highest probability next token
    prev = "sat"
    best_next = predict_most_probable(prev)
    print(f"Most probable token after '{prev}': '{best_next}'\n")

    # 3. Verify probability sum property
    verify_probability_distributions()

    # 4. Mode A: Greedy generation (5 sentences)
    print("--- Mode A: Greedy Generation ---")
    for i in range(5):
        print(f"{i+1}. {generate_sentence(mode='greedy')}")
    print()

    # 5. Mode B: Sampling generation (5 sentences)
    print("--- Mode B: Sampling Generation ---")
    for i in range(5):
        print(f"{i+1}. {generate_sentence(mode='sampling')}")
