"""
Warehouse Navigation Agent
===========================
A goal-based agent that navigates an autonomous warehouse vehicle
from a starting position (S) to a goal position (G) while avoiding
obstacles (#) in a 2D grid environment.

Search Algorithm: Breadth-First Search (BFS)
---------------------------------------------
BFS is chosen because:
1. It guarantees finding the SHORTEST path in an unweighted grid.
2. All moves (Up, Down, Left, Right) have equal cost (one grid square).
3. It systematically explores all positions at distance d before
   exploring positions at distance d+1, ensuring optimality.
4. It is complete — if a path exists, BFS will find it.

Author: AI Agents Lab Exercise
"""

from collections import deque


def parse_warehouse(warehouse_str):
    """
    Parse the warehouse string into a 2D grid and locate S and G positions.

    Parameters
    ----------
    warehouse_str : str
        Multi-line string representing the warehouse map.

    Returns
    -------
    grid : list of list of str
        2D grid representation of the warehouse.
    start : tuple (row, col)
        Position of 'S' in the grid.
    goal : tuple (row, col)
        Position of 'G' in the grid.
    """
    grid = []
    start = None
    goal = None

    for r, line in enumerate(warehouse_str.strip().split("\n")):
        row = list(line)
        grid.append(row)
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)

    return grid, start, goal


def bfs(grid, start, goal):
    """
    Perform Breadth-First Search to find the shortest path from start to goal.

    Parameters
    ----------
    grid : list of list of str
        2D grid of the warehouse.
    start : tuple (row, col)
        Starting position.
    goal : tuple (row, col)
        Goal position.

    Returns
    -------
    path : list of tuple (row, col) or None
        Ordered list of positions from start to goal, or None if no path exists.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Four possible movement directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction_names = ["Up", "Down", "Left", "Right"]

    # BFS queue stores (current_position, path_so_far)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (row, col), path = queue.popleft()

        # Check if we reached the goal
        if (row, col) == goal:
            return path

        # Explore all four neighbours
        for (dr, dc), name in zip(directions, direction_names):
            new_row, new_col = row + dr, col + dc

            # Check bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Check if the cell is free (not an obstacle) and not visited
                if grid[new_row][new_col] != "#" and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), path + [(new_row, new_col)]))

    # No path found
    return None


def get_move_directions(path):
    """
    Convert a path of positions into a sequence of human-readable move directions.

    Parameters
    ----------
    path : list of tuple (row, col)
        Ordered list of positions.

    Returns
    -------
    moves : list of str
        Sequence of direction names (Up, Down, Left, Right).
    """
    direction_map = {
        (-1, 0): "Up",
        (1, 0): "Down",
        (0, -1): "Left",
        (0, 1): "Right",
    }
    moves = []
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        moves.append(direction_map[(dr, dc)])
    return moves


def display_path_on_grid(grid, path):
    """
    Display the warehouse grid with the path marked using '*'.

    Parameters
    ----------
    grid : list of list of str
        2D grid of the warehouse.
    path : list of tuple (row, col)
        Path from start to goal.
    """
    # Create a copy of the grid
    display = [row[:] for row in grid]

    # Mark path positions (excluding start and goal)
    for r, c in path[1:-1]:
        display[r][c] = "*"

    print("\nWarehouse with path marked (* = path):")
    for row in display:
        print("".join(row))


def main():
    """Main function — run the goal-based warehouse navigation agent."""

    # ---- Environment ----
    warehouse = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

    print("=" * 50)
    print("  WAREHOUSE NAVIGATION AGENT")
    print("  Goal-Based Agent using BFS")
    print("=" * 50)

    # Parse the warehouse
    grid, start, goal = parse_warehouse(warehouse)

    print(f"\nWarehouse dimensions: {len(grid)} rows x {len(grid[0])} columns")
    print(f"Start position (S): row {start[0]}, col {start[1]}")
    print(f"Goal  position (G): row {goal[0]}, col {goal[1]}")

    print("\nOriginal warehouse map:")
    for row in grid:
        print("".join(row))

    # ---- Search for a path ----
    print("\nSearching for a collision-free path using BFS...")
    path = bfs(grid, start, goal)

    if path is None:
        print("\nNo path exists from S to G.")
        print("The goal is unreachable given the current obstacle layout.")
    else:
        print(f"\nPath found! Length: {len(path)} positions ({len(path) - 1} moves)")

        # Display the path on the grid
        display_path_on_grid(grid, path)

        # Show the move sequence
        moves = get_move_directions(path)
        print(f"\nMove sequence ({len(moves)} moves):")
        for i, move in enumerate(moves, 1):
            print(f"  Step {i:3d}: {move}")

        # Show path coordinates
        print(f"\nPath coordinates:")
        for i, (r, c) in enumerate(path):
            label = grid[r][c]
            if (r, c) == start:
                label = "S (start)"
            elif (r, c) == goal:
                label = "G (goal)"
            else:
                label = "."
            print(f"  Position {i:3d}: ({r}, {c}) [{label}]")


if __name__ == "__main__":
    main()
