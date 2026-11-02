"""Warehouse Robot Navigation Agent — Search & A*
===============================================
An intelligent goal-based agent implementing Breadth-First Search (BFS)
and A* Search with multiple heuristic functions to solve warehouse
navigation problems on 2D grid maps.

Course: Artificial Intelligence — Laboratory Exercise (Search & A*)
"""

from collections import deque
import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]


class WarehouseAgent:
  """Goal-based navigation agent operating on a 2D warehouse grid."""

  def __init__(self, raw_map: str):
    self.grid: List[str] = [
        line.strip() for line in raw_map.strip().splitlines() if line.strip()
    ]
    self.height: int = len(self.grid)
    self.width: int = len(self.grid[0])
    self.start: Coord = self._find_symbol("S")
    self.goal: Coord = self._find_symbol("G")

    # Cardinal movements: Up, Down, Left, Right
    self.actions: Dict[str, Tuple[int, int]] = {
        "Up": (-1, 0),
        "Down": (1, 0),
        "Left": (0, -1),
        "Right": (0, 1),
    }

  def _find_symbol(self, symbol: str) -> Coord:
    for r in range(self.height):
      for c in range(self.width):
        if self.grid[r][c] == symbol:
          return (r, c)
    raise ValueError(f"Symbol '{symbol}' not found in map.")

  def _heuristic(self, state: Coord, mode: str = "manhattan") -> float:
    """Calculate heuristic estimate h(n) from state to goal.

    Modes:
      - 'manhattan': |r - rG| + |c - cG| (Admissible & consistent for 4-way
      grid)
      - 'euclidean': sqrt((r - rG)^2 + (c - cG)^2) (Admissible, less informed)
      - 'zero':      0 (Uniform-Cost / Dijkstra blind search)
      - 'manhattan_x2': 2 * (|r - rG| + |c - cG|) (Inadmissible / aggressive
      weighted A*)
    """
    dr = abs(state[0] - self.goal[0])
    dc = abs(state[1] - self.goal[1])

    if mode == "manhattan":
      return float(dr + dc)
    elif mode == "euclidean":
      return math.sqrt(dr * dr + dc * dc)
    elif mode == "zero":
      return 0.0
    elif mode == "manhattan_x2":
      return 2.0 * float(dr + dc)
    else:
      raise ValueError(f"Unknown heuristic mode: '{mode}'")

  def _get_neighbors(self, state: Coord) -> List[Tuple[Coord, str]]:
    """Return all valid, collision-free neighbor states and their corresponding actions."""
    r, c = state
    neighbors = []
    for action, (dr, dc) in self.actions.items():
      nr, nc = r + dr, c + dc
      if (
          0 <= nr < self.height
          and 0 <= nc < self.width
          and self.grid[nr][nc] != "#"
      ):
        neighbors.append(((nr, nc), action))
    return neighbors

  def solve_astar(self, heuristic_mode: str = "manhattan") -> Dict:
    """Execute A* Search from start to goal using the specified heuristic."""
    counter = 0
    # Priority queue stores tuples of: (f_score, tie_breaker_counter, state)
    frontier: List[Tuple[float, int, Coord]] = []
    h_start = self._heuristic(self.start, heuristic_mode)
    heapq.heappush(frontier, (h_start, counter, self.start))

    g_score: Dict[Coord, int] = {self.start: 0}
    parent_map: Dict[Coord, Tuple[Coord, str]] = {}
    closed_set: Set[Coord] = set()
    states_expanded = 0

    while frontier:
      f_current, _, current = heapq.heappop(frontier)

      if current in closed_set:
        continue

      closed_set.add(current)
      states_expanded += 1

      # Goal test upon expansion
      if current == self.goal:
        path, actions, step_details = self._reconstruct_path(
            parent_map, current
        )
        return {
            "found": True,
            "path": path,
            "actions": actions,
            "step_details": step_details,
            "path_length": len(actions),
            "states_expanded": states_expanded,
            "heuristic": heuristic_mode,
        }

      for neighbor, action in self._get_neighbors(current):
        tentative_g = g_score[current] + 1
        if tentative_g < g_score.get(neighbor, float("inf")):
          g_score[neighbor] = tentative_g
          h_neighbor = self._heuristic(neighbor, heuristic_mode)
          f_score = tentative_g + h_neighbor
          parent_map[neighbor] = (current, action)
          counter += 1
          heapq.heappush(frontier, (f_score, counter, neighbor))

    return {
        "found": False,
        "path": [],
        "actions": [],
        "step_details": [],
        "path_length": 0,
        "states_expanded": states_expanded,
        "heuristic": heuristic_mode,
    }

  def solve_bfs(self) -> Dict:
    """Execute Breadth-First Search (BFS) from start to goal."""
    queue = deque([self.start])
    visited: Set[Coord] = {self.start}
    parent_map: Dict[Coord, Tuple[Coord, str]] = {}
    closed_set: Set[Coord] = set()
    states_expanded = 0

    while queue:
      current = queue.popleft()
      closed_set.add(current)
      states_expanded += 1

      # Goal test upon expansion (consistent with A*)
      if current == self.goal:
        path, actions, step_details = self._reconstruct_path(
            parent_map, current
        )
        return {
            "found": True,
            "path": path,
            "actions": actions,
            "step_details": step_details,
            "path_length": len(actions),
            "states_expanded": states_expanded,
            "algorithm": "BFS",
        }

      for neighbor, action in self._get_neighbors(current):
        if neighbor not in visited:
          visited.add(neighbor)
          parent_map[neighbor] = (current, action)
          queue.append(neighbor)

    return {
        "found": False,
        "path": [],
        "actions": [],
        "step_details": [],
        "path_length": 0,
        "states_expanded": states_expanded,
        "algorithm": "BFS",
    }

  def _reconstruct_path(
      self, parent_map: Dict[Coord, Tuple[Coord, str]], current: Coord
  ):
    """Backtrack from goal to start to reconstruct path coordinates and action sequence."""
    path = [current]
    actions = []
    while current in parent_map:
      parent, action = parent_map[current]
      actions.append(action)
      path.append(parent)
      current = parent

    path.reverse()
    actions.reverse()

    step_details = []
    for i in range(len(actions)):
      from_cell = path[i]
      to_cell = path[i + 1]
      act = actions[i]
      step_details.append({
          "step": i + 1,
          "from": from_cell,
          "action": act,
          "to": to_cell,
      })

    return path, actions, step_details

  def display_solution(self, path: List[Coord]) -> None:
    """Print ASCII representation of warehouse with the solution path marked with '*'."""
    path_set = set(path)
    grid_copy = [list(row) for row in self.grid]
    for r, c in path_set:
      if grid_copy[r][c] not in ("S", "G"):
        grid_copy[r][c] = "*"

    for row in grid_copy:
      print("".join(row))


def run_laboratory_experiments():
  """Run all test suites, algorithm comparisons, and heuristic investigations."""

  # 1. Original Warehouse Map
  warehouse_map = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""

  agent = WarehouseAgent(warehouse_map)
  result_astar = agent.solve_astar(heuristic_mode="manhattan")

  print("=" * 65)
  print("  TASK 3 - TEST 1: ORIGINAL WAREHOUSE (A* WITH MANHATTAN)")
  print("=" * 65)
  print(f"Solution Found:   {result_astar['found']}")
  print(f"Path Length:      {result_astar['path_length']} steps")
  print(f"States Expanded:  {result_astar['states_expanded']}")
  print(f"Start Position:   {agent.start} (Row {agent.start[0]}, Col {agent.start[1]})")
  print(f"Goal Position:    {agent.goal} (Row {agent.goal[0]}, Col {agent.goal[1]})")
  print("-" * 65)
  print("Full Coordinate Path [ (row, col) ]:")
  print(" -> ".join([f"({r},{c})" for r, c in result_astar["path"]]))
  print("-" * 65)
  print("Map Visualisation (* = Path taken):")
  agent.display_solution(result_astar["path"])

  # 2. Additional Task 3 Systematic Tests
  print("\n" + "=" * 65)
  print("  TASK 3: SYSTEMATIC TESTS (TESTS 2, 3, 4)")
  print("=" * 65)

  # Test 2: Trivial Case
  trivial_map = """
#####
#SG##
#####
"""
  agent_trivial = WarehouseAgent(trivial_map)
  res_t2 = agent_trivial.solve_astar()
  print(f"\n[Test 2: Trivial Case (Adjacent S and G)]")
  print(f"  Map:\n{trivial_map.strip()}")
  print(f"  Solution Found:   {res_t2['found']}")
  print(f"  Path:             {res_t2['path']}")
  print(f"  Path Length:      {res_t2['path_length']} step")
  print(f"  States Expanded:  {res_t2['states_expanded']}")

  # Test 3: No Solution Case
  no_sol_map = """
#######
#S....#
###.###
#...#G#
#######
"""
  agent_nosol = WarehouseAgent(no_sol_map)
  res_t3 = agent_nosol.solve_astar()
  print(f"\n[Test 3: No Solution (Inaccessible Goal)]")
  print(f"  Map:\n{no_sol_map.strip()}")
  print(f"  Solution Found:   {res_t3['found']}")
  print(f"  Path Length:      {res_t3['path_length']}")
  print(f"  States Expanded:  {res_t3['states_expanded']} (Terminated safely without infinite loop)")

  # Test 4: Alternative Paths Case
  alt_map = """
###########
#S.......G#
#.#######.#
#.........#
###########
"""
  agent_alt = WarehouseAgent(alt_map)
  res_t4 = agent_alt.solve_astar()
  print(f"\n[Test 4: Alternative Paths (Direct 8-step vs 16-step Detour)]")
  print(f"  Map:\n{alt_map.strip()}")
  print(f"  Solution Found:   {res_t4['found']}")
  print(f"  Path Length:      {res_t4['path_length']} steps (Optimal path returned)")
  print(f"  States Expanded:  {res_t4['states_expanded']}")

  # 3. Task 5: BFS vs A* Comparison
  print("\n" + "=" * 65)
  print("  TASK 5: BFS VS A* COMPARISON ON ORIGINAL WAREHOUSE")
  print("=" * 65)
  result_bfs = agent.solve_bfs()

  print(f"{'Measure':<20} | {'BFS':<15} | {'A* (Manhattan)':<15}")
  print("-" * 56)
  print(f"{'Solution found':<20} | {str(result_bfs['found']):<15} | {str(result_astar['found']):<15}")
  print(f"{'Path length':<20} | {result_bfs['path_length']:<15} | {result_astar['path_length']:<15}")
  print(f"{'States expanded':<20} | {result_bfs['states_expanded']:<15} | {result_astar['states_expanded']:<15}")

  # 4. Task 6: Heuristic Investigation
  print("\n" + "=" * 65)
  print("  TASK 6: HEURISTIC INVESTIGATION")
  print("=" * 65)

  h_modes = [
      ("Zero (h=0)", "zero"),
      ("Euclidean", "euclidean"),
      ("Manhattan", "manhattan"),
      ("2 x Manhattan", "manhattan_x2"),
  ]

  print("\n--- On Original Warehouse Map ---")
  print(f"{'Heuristic':<20} | {'Found':<8} | {'Path Length':<12} | {'States Expanded':<15}")
  print("-" * 62)
  for label, mode in h_modes:
    res = agent.solve_astar(heuristic_mode=mode)
    print(f"{label:<20} | {str(res['found']):<8} | {res['path_length']:<12} | {res['states_expanded']:<15}")

  print("\n--- On Alternative Paths Map (Showing Pruning Capability) ---")
  print(f"{'Heuristic':<20} | {'Found':<8} | {'Path Length':<12} | {'States Expanded':<15}")
  print("-" * 62)
  for label, mode in h_modes:
    res = agent_alt.solve_astar(heuristic_mode=mode)
    print(f"{label:<20} | {str(res['found']):<8} | {res['path_length']:<12} | {res['states_expanded']:<15}")


if __name__ == "__main__":
  run_laboratory_experiments()
