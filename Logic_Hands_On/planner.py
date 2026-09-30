"""
Preconditions −→ When is an action applicable?
Effects −→ How does the state change?
Goal −→ When does planning terminate?
BFS −→ How are alternative plans explored?

Preconditions: is_applicable() checks this, by ensuring that pos_preconditions is a subset of state, and neg_preconditions intersection state is empty
Effects: apply() function does this
Goal:
if goal.issubset(initial_state):
    return [], [initial_state]

Task 4: Check task applicability
logic decides what a single step is allowed to do, 
given precisely what's true right now. Search decides 
which sequence of legal steps to try, since usually many
different actions are applicable at once and only some 
sequences reach the goal. Logic without search would just 
tell you what's applicable right now, 
with no way to look ahead. Search without logic would have 
no way to know which moves are even legal.

LOGIC: What's allowed in a single step
SEARCH: Chains of steps are checked


"""


"""
Simple STRIPS-style planning agent using breadth-first search.

State representation: a state is a frozenset of string propositions,
e.g. frozenset({"At(Robot,A)", "At(Package,A)"})

Each Action has:
  - name
  - pos_pre: propositions that must be TRUE for the action to apply
  - neg_pre: propositions that must be FALSE for the action to apply
  - pos_eff: propositions added to the state after the action
  - neg_eff: propositions removed from the state after the action
"""

from collections import deque


class Action:
    def __init__(self, name, pos_pre=None, neg_pre=None, pos_eff=None, neg_eff=None):
        self.name = name
        self.pos_pre = frozenset(pos_pre or [])
        self.neg_pre = frozenset(neg_pre or [])
        self.pos_eff = frozenset(pos_eff or [])
        self.neg_eff = frozenset(neg_eff or [])

    def is_applicable(self, state):
        # S |= Preconditions(a)
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        # remove negative effects, then add positive effects
        return (state - self.neg_eff) | self.pos_eff

    def __repr__(self):
        return self.name


def bfs_plan(initial_state, goal, actions, verbose=True):
    """
    Breadth-first search over the state space.
    Returns (plan, trace) where plan is a list of Actions and
    trace is the list of states visited (S0, S1, ..., Sn).
    Returns (None, None) if no plan exists.
    """
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal.issubset(initial_state):
        return [], [initial_state]

    visited = {initial_state}
    # queue holds (state, plan_so_far, trace_so_far)
    queue = deque([(initial_state, [], [initial_state])])

    while queue:
        state, plan, trace = queue.popleft()

        for action in actions:
            if not action.is_applicable(state):
                continue
            new_state = action.apply(state)
            if new_state in visited:
                continue

            new_plan = plan + [action]
            new_trace = trace + [new_state]

            if goal.issubset(new_state):
                if verbose:
                    print("Plan found:")
                    for a in new_plan:
                        print(" ", a.name)
                return new_plan, new_trace

            visited.add(new_state)
            queue.append((new_state, new_plan, new_trace))

    if verbose:
        print("No plan found")
    return None, None


def print_trace(trace):
    for i, state in enumerate(trace):
        print(f"S{i}:", ", ".join(sorted(state)))


# ---------------------------------------------------------------------
# Warehouse problem setup
# ---------------------------------------------------------------------

def make_warehouse_actions(include_pickup=True):
    actions = []
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
    for x, y in connections:
        actions.append(Action(
            f"Move(Robot,{x},{y})",
            pos_pre=[f"At(Robot,{x})"],
            neg_pre=[],
            pos_eff=[f"At(Robot,{y})"],
            neg_eff=[f"At(Robot,{x})"],
        ))

    if include_pickup:
        for loc in ["A", "B", "C"]:
            actions.append(Action(
                f"PickUp(Package,{loc})",
                pos_pre=[f"At(Robot,{loc})", f"At(Package,{loc})"],
                neg_pre=[],
                pos_eff=["Holding(Package)"],
                neg_eff=[f"At(Package,{loc})"],
            ))
            actions.append(Action(
                f"Drop(Package,{loc})",
                pos_pre=[f"At(Robot,{loc})", "Holding(Package)"],
                neg_pre=[],
                pos_eff=[f"At(Package,{loc})"],
                neg_eff=["Holding(Package)"],
            ))
    return actions


if __name__ == "__main__":
    initial = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}

    print("=== TEST A: Solvable problem ===")
    actions_a = make_warehouse_actions(include_pickup=True)
    plan_a, trace_a = bfs_plan(initial, goal, actions_a)
    if trace_a:
        print_trace(trace_a)

    print("\n=== TEST B: Impossible problem (no PickUp action) ===")
    actions_b = make_warehouse_actions(include_pickup=False)
    plan_b, trace_b = bfs_plan(initial, goal, actions_b)

    print("\n=== TEST C: Irrelevant actions (robot reaches C, package doesn't) ===")
    # Same actions as Test A, but check a "trap" goal that only requires the
    # robot (not the package) to be at C, to confirm the planner distinguishes them.
    trap_goal = {"At(Robot,C)"}
    plan_c, trace_c = bfs_plan(initial, trap_goal, actions_a)
    if trace_c:
        print_trace(trace_c)
    print("\nNote: reaching At(Robot,C) does NOT satisfy At(Package,C).")
    print("Package position in final state of Test A's plan:",
          [f for f in trace_a[-1] if "Package" in f])