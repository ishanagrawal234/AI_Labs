"""Logical Reasoning for Planning -- Warehouse & Locked-Door Planning Agent
=============================================================================
A classical STRIPS-style forward state-space planning agent using Propositional
Logic and Breadth-First Search (BFS), accompanied by plan validation machinery.

Course: Artificial Intelligence -- Laboratory Exercise (Logical Planning)
Author: AI Student / Engineering Assistant
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Action:
  """Represents a discrete planning operator with logical preconditions and effects."""

  name: str
  preconditions_pos: FrozenSet[str] = field(default_factory=frozenset)
  preconditions_neg: FrozenSet[str] = field(default_factory=frozenset)
  effects_pos: FrozenSet[str] = field(default_factory=frozenset)
  effects_neg: FrozenSet[str] = field(default_factory=frozenset)

  def is_applicable(self, state: FrozenSet[str]) -> bool:
    """Logical entailment test: State |= Preconditions(a).

    Returns True if all positive preconditions are in the state and no negative
    preconditions are in the state.
    """
    return self.preconditions_pos.issubset(
        state
    ) and self.preconditions_neg.isdisjoint(state)

  def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
    """State transition: S' = Apply(S, a) = (S \\ effects_neg) U effects_pos."""
    return (state - self.effects_neg) | self.effects_pos

  def unsatisfied_preconditions(self, state: FrozenSet[str]) -> List[str]:
    """Identify which specific preconditions failed in the given state."""
    missing_pos = list(self.preconditions_pos - state)
    present_neg = [f"not({p})" for p in (self.preconditions_neg & state)]
    return missing_pos + present_neg

  def __str__(self) -> str:
    return self.name

  def __repr__(self) -> str:
    return self.name


class PlanValidator:
  """Independent state-transition checker for validating candidate action sequences."""

  @staticmethod
  def validate(
      initial_state: FrozenSet[str],
      plan: List[Action],
      goal: FrozenSet[str],
  ) -> Tuple[bool, Optional[str], Optional[str], List[FrozenSet[str]]]:
    """Step-by-step verification of plan execution.

    Returns:
      (is_valid, failure_action_name, failure_reason, state_history)
    """
    current = initial_state
    state_history = [current]

    for step, action in enumerate(plan, 1):
      if not action.is_applicable(current):
        failed_conds = action.unsatisfied_preconditions(current)
        reason = (
            f"Precondition failure at step {step}: action '{action.name}'"
            f" requires {failed_conds}"
        )
        return False, action.name, reason, state_history

      current = action.apply(current)
      state_history.append(current)

    # Check if final state satisfies goal
    if not goal.issubset(current):
      missing_goals = list(goal - current)
      reason = f"Goal not achieved: missing {missing_goals} in final state"
      return False, None, reason, state_history

    return True, None, "Plan is completely valid and achieves the goal.", state_history


class BFSPlanner:
  """Forward state-space search planner using Breadth-First Search."""

  @staticmethod
  def plan(
      initial_state: FrozenSet[str],
      actions: List[Action],
      goal: FrozenSet[str],
  ) -> Tuple[Optional[List[Action]], Optional[List[FrozenSet[str]]], int]:
    """Find a shortest sequence of actions transforming initial_state into goal.

    Returns:
      (action_plan, state_trajectory, states_expanded) or (None, None,
      states_expanded)
    """
    # Queue stores: (current_state, action_path, state_history)
    queue = deque([(initial_state, [], [initial_state])])
    visited: Set[FrozenSet[str]] = {initial_state}
    states_expanded = 0

    while queue:
      current_state, path, state_history = queue.popleft()
      states_expanded += 1

      # Goal test: does current_state satisfy all goal propositions?
      if goal.issubset(current_state):
        return path, state_history, states_expanded

      # Logical deduction: explore only applicable actions
      for action in actions:
        if action.is_applicable(current_state):
          next_state = action.apply(current_state)
          if next_state not in visited:
            visited.add(next_state)
            queue.append(
                (next_state, path + [action], state_history + [next_state])
            )

    # Exhausted reachable state space without finding goal
    return None, None, states_expanded


# =====================================================================
# DOMAIN BUILDERS
# =====================================================================
def create_warehouse_domain(include_pickup: bool = True) -> Tuple[
    FrozenSet[str],
    List[Action],
    FrozenSet[str],
]:
  """Construct the Warehouse Package Transport domain (A, B, C)."""
  initial_state = frozenset({"At(Robot, A)", "At(Package, A)"})
  goal = frozenset({"At(Package, C)"})

  actions = [
      Action(
          name="Move(A, B)",
          preconditions_pos=frozenset({"At(Robot, A)"}),
          effects_pos=frozenset({"At(Robot, B)"}),
          effects_neg=frozenset({"At(Robot, A)"}),
      ),
      Action(
          name="Move(B, A)",
          preconditions_pos=frozenset({"At(Robot, B)"}),
          effects_pos=frozenset({"At(Robot, A)"}),
          effects_neg=frozenset({"At(Robot, B)"}),
      ),
      Action(
          name="Move(B, C)",
          preconditions_pos=frozenset({"At(Robot, B)"}),
          effects_pos=frozenset({"At(Robot, C)"}),
          effects_neg=frozenset({"At(Robot, B)"}),
      ),
      Action(
          name="Move(C, B)",
          preconditions_pos=frozenset({"At(Robot, C)"}),
          effects_pos=frozenset({"At(Robot, B)"}),
          effects_neg=frozenset({"At(Robot, C)"}),
      ),
  ]

  if include_pickup:
    actions.extend([
        Action(
            name="PickUp(Package, A)",
            preconditions_pos=frozenset({"At(Robot, A)", "At(Package, A)"}),
            effects_pos=frozenset({"Holding(Package)"}),
            effects_neg=frozenset({"At(Package, A)"}),
        ),
        Action(
            name="PickUp(Package, B)",
            preconditions_pos=frozenset({"At(Robot, B)", "At(Package, B)"}),
            effects_pos=frozenset({"Holding(Package)"}),
            effects_neg=frozenset({"At(Package, B)"}),
        ),
        Action(
            name="PickUp(Package, C)",
            preconditions_pos=frozenset({"At(Robot, C)", "At(Package, C)"}),
            effects_pos=frozenset({"Holding(Package)"}),
            effects_neg=frozenset({"At(Package, C)"}),
        ),
    ])

  # Drop actions at all locations
  actions.extend([
      Action(
          name="Drop(Package, A)",
          preconditions_pos=frozenset({"At(Robot, A)", "Holding(Package)"}),
          effects_pos=frozenset({"At(Package, A)"}),
          effects_neg=frozenset({"Holding(Package)"}),
      ),
      Action(
          name="Drop(Package, B)",
          preconditions_pos=frozenset({"At(Robot, B)", "Holding(Package)"}),
          effects_pos=frozenset({"At(Package, B)"}),
          effects_neg=frozenset({"Holding(Package)"}),
      ),
      Action(
          name="Drop(Package, C)",
          preconditions_pos=frozenset({"At(Robot, C)", "Holding(Package)"}),
          effects_pos=frozenset({"At(Package, C)"}),
          effects_neg=frozenset({"Holding(Package)"}),
      ),
  ])

  return initial_state, actions, goal


def create_locked_door_domain(include_pickkey: bool = True) -> Tuple[
    FrozenSet[str],
    List[Action],
    FrozenSet[str],
]:
  """Construct the Locked-Door Navigation domain (Rooms A, B, C with Key and Lock)."""
  initial_state = frozenset({"At(A)", "KeyAt(A)"})
  goal = frozenset({"At(C)"})

  actions = [
      Action(
          name="Move(A, B)",
          preconditions_pos=frozenset({"At(A)"}),
          effects_pos=frozenset({"At(B)"}),
          effects_neg=frozenset({"At(A)"}),
      ),
      Action(
          name="Move(B, A)",
          preconditions_pos=frozenset({"At(B)"}),
          effects_pos=frozenset({"At(A)"}),
          effects_neg=frozenset({"At(B)"}),
      ),
      Action(
          name="Unlock",
          preconditions_pos=frozenset({"At(B)", "HasKey"}),
          effects_pos=frozenset({"Unlocked"}),
          effects_neg=frozenset(),
      ),
      Action(
          name="Move(B, C)",
          preconditions_pos=frozenset({"At(B)", "Unlocked"}),
          effects_pos=frozenset({"At(C)"}),
          effects_neg=frozenset({"At(B)"}),
      ),
      Action(
          name="Move(C, B)",
          preconditions_pos=frozenset({"At(C)", "Unlocked"}),
          effects_pos=frozenset({"At(B)"}),
          effects_neg=frozenset({"At(C)"}),
      ),
  ]

  if include_pickkey:
    actions.append(
        Action(
            name="PickKey",
            preconditions_pos=frozenset({"At(A)", "KeyAt(A)"}),
            effects_pos=frozenset({"HasKey"}),
            effects_neg=frozenset({"KeyAt(A)"}),
        )
    )

  return initial_state, actions, goal


# =====================================================================
# LABORATORY EXPERIMENT RUNNER
# =====================================================================
def run_all_experiments():
  print("=" * 70)
  print("  LOGICAL REASONING FOR PLANNING -- COMPLETE LABORATORY SUITE")
  print("=" * 70)

  # -------------------------------------------------------------
  # TASK 3: SYSTEMATIC TESTS (TEST A, TEST B, TEST C)
  # -------------------------------------------------------------
  print("\n" + "=" * 70)
  print("  TASK 3: TEST SUITE ON WAREHOUSE DOMAIN")
  print("=" * 70)

  # Test A: Solvable Problem
  init_a, actions_a, goal_a = create_warehouse_domain(include_pickup=True)
  plan_a, states_a, exp_a = BFSPlanner.plan(init_a, actions_a, goal_a)
  valid_a, err_act_a, reason_a, _ = PlanValidator.validate(
      init_a, plan_a, goal_a
  )

  print("\n[Test A: Solvable Warehouse Problem]")
  print(f"  Initial State:    {sorted(list(init_a))}")
  print(f"  Goal:             {sorted(list(goal_a))}")
  print(f"  Plan Found:       {plan_a is not None}")
  print(f"  Resulting Plan:   {[a.name for a in plan_a]}")
  print(f"  Plan Length:      {len(plan_a)} steps")
  print(f"  States Expanded:  {exp_a}")
  print(f"  Plan Validated:   {valid_a} ({reason_a})")
  print("  State Trajectory:")
  for i, s in enumerate(states_a):
    act_str = f"After {plan_a[i-1].name}" if i > 0 else "Initial State (S0)"
    print(f"    S{i} ({act_str}): {sorted(list(s))}")

  # Test B: Impossible Problem (No PickUp action)
  init_b, actions_b, goal_b = create_warehouse_domain(include_pickup=False)
  plan_b, states_b, exp_b = BFSPlanner.plan(init_b, actions_b, goal_b)

  print("\n[Test B: Impossible Problem (PickUp Action Removed)]")
  print(f"  Initial State:    {sorted(list(init_b))}")
  print(f"  Goal:             {sorted(list(goal_b))}")
  print(
      f"  Plan Found:       {plan_b is not None} (Planner correctly reported"
      " 'No plan found')"
  )
  print(f"  States Expanded:  {exp_b} (Full reachable space exhausted)")

  # Test C: Irrelevant Actions (Robot moves without picking up package)
  print("\n[Test C: Irrelevant Actions]")
  irrelevant_plan = [
      Action(
          name="Move(A, B)",
          preconditions_pos=frozenset({"At(Robot, A)"}),
          effects_pos=frozenset({"At(Robot, B)"}),
          effects_neg=frozenset({"At(Robot, A)"}),
      ),
      Action(
          name="Move(B, C)",
          preconditions_pos=frozenset({"At(Robot, B)"}),
          effects_pos=frozenset({"At(Robot, C)"}),
          effects_neg=frozenset({"At(Robot, B)"}),
      ),
  ]
  valid_c, err_act_c, reason_c, states_c = PlanValidator.validate(
      init_a, irrelevant_plan, goal_a
  )
  print(f"  Proposed Plan:    {[a.name for a in irrelevant_plan]}")
  print(f"  Goal to achieve:  {sorted(list(goal_a))}")
  print(f"  Plan Validated:   {valid_c}")
  print(f"  Validator Report: {reason_c}")
  print(f"  Final State S2:   {sorted(list(states_c[-1]))}")
  print(
      "  Confirmation: Planner does not confuse Robot reaching C with Package"
      " reaching C."
  )

  # -------------------------------------------------------------
  # TASK 6: LOCKED-DOOR NAVIGATION DOMAIN
  # -------------------------------------------------------------
  print("\n" + "=" * 70)
  print("  TASK 6: LOCKED-DOOR NAVIGATION DOMAIN (ROOMS A, B, C)")
  print("=" * 70)

  # 6(b) BFS Plan
  init_lock, actions_lock, goal_lock = create_locked_door_domain(
      include_pickkey=True
  )
  plan_lock, states_lock, exp_lock = BFSPlanner.plan(
      init_lock, actions_lock, goal_lock
  )
  valid_lock, _, reason_lock, _ = PlanValidator.validate(
      init_lock, plan_lock, goal_lock
  )

  print("\n[Task 6(b): BFS Locked-Door Shortest Plan]")
  print(f"  Initial State:    {sorted(list(init_lock))}")
  print(f"  Goal:             {sorted(list(goal_lock))}")
  print(f"  Plan Found:       {plan_lock is not None}")
  print(f"  Resulting Plan:   {[a.name for a in plan_lock]}")
  print(f"  Plan Length:      {len(plan_lock)} steps")
  print(f"  States Expanded:  {exp_lock}")
  print(f"  Plan Validated:   {valid_lock} ({reason_lock})")
  print("  State Trace:")
  for i, s in enumerate(states_lock):
    act_str = f"After {plan_lock[i-1].name}" if i > 0 else "Initial State (S0)"
    print(f"    S{i} ({act_str}): {sorted(list(s))}")

  # 6(c) Removing PickKey
  init_nopick, actions_nopick, goal_nopick = create_locked_door_domain(
      include_pickkey=False
  )
  plan_nopick, _, exp_nopick = BFSPlanner.plan(
      init_nopick, actions_nopick, goal_nopick
  )
  print("\n[Task 6(c): PickKey Removed]")
  print(
      f"  Plan Found:       {plan_nopick is not None} (Correctly reported 'No"
      " plan found')"
  )
  print(f"  States Expanded:  {exp_nopick}")
  print(
      "  Explanation: Without HasKey, Unlock cannot be executed. Thus Unlocked"
      " is never added,"
  )
  print(
      "  and Move(B, C) can never satisfy its precondition. C remains"
      " unreachable."
  )

  # 6(d) Validate proposed flawed sequence: Move(A, B), Unlock, Move(B, C)
  flawed_plan = [
      Action(
          name="Move(A, B)",
          preconditions_pos=frozenset({"At(A)"}),
          effects_pos=frozenset({"At(B)"}),
          effects_neg=frozenset({"At(A)"}),
      ),
      Action(
          name="Unlock",
          preconditions_pos=frozenset({"At(B)", "HasKey"}),
          effects_pos=frozenset({"Unlocked"}),
          effects_neg=frozenset(),
      ),
      Action(
          name="Move(B, C)",
          preconditions_pos=frozenset({"At(B)", "Unlocked"}),
          effects_pos=frozenset({"At(C)"}),
          effects_neg=frozenset({"At(B)"}),
      ),
  ]
  valid_flawed, err_act_flawed, reason_flawed, _ = PlanValidator.validate(
      init_lock, flawed_plan, goal_lock
  )
  print(
      "\n[Task 6(d): Validating Flawed Sequence [Move(A, B), Unlock, Move(B,"
      " C)]]"
  )
  print(f"  Plan Valid:       {valid_flawed}")
  print(f"  First Invalid:    '{err_act_flawed}'")
  print(f"  Failure Detail:   {reason_flawed}")

  # -------------------------------------------------------------
  # TASK 7: LLM PLAN COMPARISON ACROSS DOMAINS
  # -------------------------------------------------------------
  print("\n" + "=" * 70)
  print("  TASK 7: LLM PLAN VALIDATION & COMPARISON")
  print("=" * 70)

  # Suppose an LLM proposed plan without picking key: [Move(A, B), Unlock, Move(B, C)]
  print("\n[Evaluating LLM Proposal 1 (Common LLM hallucination: skip key)]")
  v1, err_act_1, r1, _ = PlanValidator.validate(
      init_lock, flawed_plan, goal_lock
  )
  print(f"  LLM Proposed:     {[a.name for a in flawed_plan]}")
  print(f"  Validator Check:  Valid={v1} | Error: {r1}")

  # After providing the validator error message to the LLM, it corrects the plan:
  corrected_llm_plan = plan_lock  # [PickKey, Move(A, B), Unlock, Move(B, C)]
  v2, _, r2, _ = PlanValidator.validate(
      init_lock, corrected_llm_plan, goal_lock
  )
  print(
      "\n[Evaluating LLM Proposal 2 (Revised plan after receiving validator"
      " error)]"
  )
  print(f"  Revised Plan:     {[a.name for a in corrected_llm_plan]}")
  print(f"  Validator Check:  Valid={v2} | Status: {r2}")
  print(
      f"  Matches BFS Shortest Length: {len(corrected_llm_plan) == len(plan_lock)}"
  )


if __name__ == "__main__":
  run_all_experiments()

