"""
Tabular Q-Learning for Warehouse Robot Navigation (5x5 Grid) - CS F407
"""
import numpy as np
import matplotlib.pyplot as plt

OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 0:Up, 1:Down, 2:Left, 3:Right, 4:Wait
ACT_NAMES = ["Up", "Down", "Left", "Right", "Wait"]


class WarehouseEnv:
    """5x5 Grid warehouse with doorway at (2,2), start (2,0), goal (2,4)."""
    def __init__(self, start=(2, 0), goal=(2, 4), max_steps=40):
        self.start, self.goal, self.max_steps = start, goal, max_steps
        self.reset()

    def reset(self, pos=None):
        self.pos = self.start if pos is None else pos
        self.steps = 0
        return self.pos[0] * 5 + self.pos[1], self.pos

    def step(self, a):
        self.steps += 1
        dr, dc = ACTIONS[a]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        invalid = not (0 <= nr < 5 and 0 <= nc < 5) or (nr, nc) in OBSTACLES
        next_pos = self.pos if invalid else (nr, nc)

        # Mutually exclusive rewards
        if next_pos == self.goal:
            r, term = 10.0, True
        elif invalid:
            r, term = -1.0, False
        else:
            r, term = -0.1, False

        self.pos = next_pos
        trunc = (self.steps >= self.max_steps) and not term
        return next_pos[0] * 5 + next_pos[1], next_pos, r, term, trunc, invalid


def epsilon_greedy(q, s, eps, rng):
    """Epsilon-greedy with random tie-breaking among best actions."""
    if rng.random() < eps:
        return rng.integers(0, 5)
    return rng.choice(np.flatnonzero(q[s] == np.max(q[s])))


def q_learning_update(q, s, a, r, s_next, term, alpha=0.2, gamma=0.95):
    """One Q-learning TD update. Bootstraps at truncation (term=False)."""
    target = r if term else r + gamma * np.max(q[s_next])
    q[s, a] += alpha * (target - q[s, a])
    return target


def train_q_learning(episodes=1500, alpha=0.2, gamma=0.95, seed=7):
    """Trains Q-learning agent and displays reward statistics across iterations."""
    rng = np.random.default_rng(seed)
    env = WarehouseEnv()
    q = np.zeros((25, 5))
    returns, successes, eps_hist = [], [], []

    print(f"\n{'Episode Block':^17} | {'Avg Reward':^12} | {'Success Rate':^14} | {'Epsilon':^9}")
    print("-" * 60)

    for ep in range(episodes):
        eps = max(0.05, 1.0 - (1.0 - 0.05) * (ep / 1200))
        s, _ = env.reset()
        ep_r, done = 0.0, False

        while not done:
            a = epsilon_greedy(q, s, eps, rng)
            s_next, _, r, term, trunc, _ = env.step(a)
            q_learning_update(q, s, a, r, s_next, term, alpha, gamma)
            ep_r += r
            s = s_next
            done = term or trunc

        returns.append(ep_r)
        successes.append(1 if term else 0)
        eps_hist.append(eps)

        # Print statistics every 150 episodes
        if (ep + 1) % 150 == 0:
            window = slice(ep - 149, ep + 1)
            avg_r = np.mean(returns[window])
            succ_pct = np.mean(successes[window]) * 100
            print(f"{ep - 148:>6} - {ep + 1:>5}   |   {avg_r:>8.2f}   |    {succ_pct:>6.1f}%     |  {eps:>6.3f}")

    return q, {"returns": np.array(returns), "successes": np.array(successes), "epsilons": np.array(eps_hist)}


def evaluate_greedy(env, q, start=(2, 0)):
    """Greedy deterministic evaluation (epsilon=0, no updates)."""
    s, pos = env.reset(start)
    traj, actions, total_r = [pos], [], 0.0
    while True:
        a = int(np.argmax(q[s]))
        actions.append(a)
        s_next, pos, r, term, trunc, _ = env.step(a)
        total_r += r
        traj.append(pos)
        s = s_next
        if term or trunc:
            return traj, actions, total_r, term


def test_action(pos, action):
    """Test mode: Inspect single step outcome from any position without training."""
    env = WarehouseEnv()
    env.reset(pos)
    a = action if isinstance(action, int) else ACT_NAMES.index(action.capitalize())
    s_next, p_next, r, term, trunc, inv = env.step(a)
    return {"pos": pos, "action": ACT_NAMES[a], "next_pos": p_next, "reward": r, "term": term, "invalid": inv}


def plot_curves(history, save_path="/home/arxalier/Testing/CS_F407/Labs/RL_Hands_On/q_learning_curves.png", w=50):
    """Plot moving average training curves."""
    r_ma = np.convolve(history["returns"], np.ones(w) / w, mode="valid")
    s_ma = np.convolve(history["successes"], np.ones(w) / w, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(history["returns"], alpha=0.15, color="royalblue")
    ax1.plot(range(w - 1, len(history["returns"])), r_ma, color="royalblue", lw=2, label=f"Return (MA {w})")
    ax1.set_ylabel("Reward / Return")
    ax1.set_title("Q-Learning Reward Progression Over Episodes")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(range(w - 1, len(history["successes"])), s_ma * 100, color="forestgreen", lw=2, label=f"Success % (MA {w})")
    ax2.plot(history["epsilons"] * 100, color="crimson", lw=1.5, linestyle=":", label="Epsilon %")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Episode")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # 1. Quick test mode demonstration
    print("--- Test Mode Inspection ---")
    sample_test = test_action(pos=(1, 1), action="Right")
    print(f"Action Test: {sample_test['pos']} + {sample_test['action']} -> {sample_test['next_pos']} (Reward: {sample_test['reward']}, Invalid: {sample_test['invalid']})")

    # 2. Train and show reward increasing across iterations
    q_table, history = train_q_learning(episodes=1500)
    plot_curves(history)

    # 3. Greedy Evaluation
    traj, acts, total_r, success = evaluate_greedy(WarehouseEnv(), q_table)
    print("\n--- Greedy Evaluation Result ---")
    print(f"Success: {success} | Steps: {len(acts)} | Total Reward: {total_r:.2f}")
    print(f"Actions: {' -> '.join(ACT_NAMES[a] for a in acts)}")
    print(f"Trajectory: {' -> '.join(str(p) for p in traj)}")
