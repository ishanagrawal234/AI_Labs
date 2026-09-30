import heapq
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]


class WarehouseAgent:

  def __init__(self, raw_map: str):
    self.grid: List[str] = [
        line.strip() for line in raw_map.strip().splitlines() if line.strip()
    ]
    self.height: int = len(self.grid)
    self.width: int = len(self.grid[0])
    self.start: Coord = self._find_symbol("S")
    self.goal: Coord = self._find_symbol("G")

    # Cardinal movements: (dr, dc)
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

  def _heuristic(self, state: Coord) -> int:
    """Manhattan distance heuristic."""
    return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

  def _get_neighbors(self, state: Coord) -> List[Tuple[Coord, str]]:
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

  def solve_astar(self) -> Dict:
    counter = 0
    frontier: List[Tuple[int, int, Coord]] = []
    heapq.heappush(
        frontier, (self._heuristic(self.start), counter, self.start)
    )

    g_score: Dict[Coord, int] = {self.start: 0}
    parent_map: Dict[Coord, Tuple[Coord, str]] = {}
    closed_set: Set[Coord] = set()
    states_expanded = 0

    while frontier:
      _, _, current = heapq.heappop(frontier)

      if current in closed_set:
        continue

      closed_set.add(current)
      states_expanded += 1

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
        }

      for neighbor, action in self._get_neighbors(current):
        tentative_g = g_score[current] + 1
        if tentative_g < g_score.get(neighbor, float("inf")):
          g_score[neighbor] = tentative_g
          f_score = tentative_g + self._heuristic(neighbor)
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
    }

  def _reconstruct_path(
      self, parent_map: Dict[Coord, Tuple[Coord, str]], current: Coord
  ):
    path = [current]
    actions = []
    while current in parent_map:
      parent, action = parent_map[current]
      actions.append(action)
      path.append(parent)
      current = parent

    path.reverse()
    actions.reverse()

    # Build step-by-step transition records
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
    path_set = set(path)
    grid_copy = [list(row) for row in self.grid]
    for r, c in path_set:
      if grid_copy[r][c] not in ("S", "G"):
        grid_copy[r][c] = "*"

    for row in grid_copy:
      print("".join(row))


# --- Execution ---
if __name__ == "__main__":
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
  result = agent.solve_astar()

  print("=" * 60)
  print("A* SEARCH REPORT")
  print("=" * 60)
  print(f"Solution Found:   {result['found']}")
  print(f"Path Length:      {result['path_length']} steps")
  print(f"States Expanded:  {result['states_expanded']}")
  print(
      f"Start Position:   {agent.start} (Row {agent.start[0]}, Col"
      f" {agent.start[1]})"
  )
  print(
      f"Goal Position:    {agent.goal} (Row {agent.goal[0]}, Col"
      f" {agent.goal[1]})"
  )
  print("-" * 60)

  print("Full Coordinate Path [ (row, col) ]:")
  print(" -> ".join([f"({r},{c})" for r, c in result["path"]]))

  print("-" * 60)
  print("Step-by-Step Trajectory:")
  print(f"{'Step':<6} | {'From (r, c)':<14} | {'Action':<8} | {'To (r, c)':<14}")
  print("-" * 60)
  for item in result["step_details"]:
    from_str = f"({item['from'][0]}, {item['from'][1]})"
    to_str = f"({item['to'][0]}, {item['to'][1]})"
    print(
        f"{item['step']:<6} | {from_str:<14} | {item['action']:<8} |"
        f" {to_str:<14}"
    )

  print("-" * 60)
  print("Map Visualisation (* = Path taken):")
  agent.display_solution(result["path"])