from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass

"""
1. What is the environment?
Blocked states, free states, S, G
2. What is the goal?
Finding a path from S to G, preferably the shortest one
3. What actions are available to the agent?
{U,D,L,R}
4. Available info: {currX, currY, environment}

5. Why is it goal based?
It does not make fixed hardcoded moves based on current state

**Reflex might also not work because

# #####
#     #
#  #  
#     #
#######


Suppose the warehouse becomes twice as large.
Would the same search strategy still be appropriate?
What additional difficulties might arise?

It might be beneficial to switch to an algorithm (heuristic based, informed search) such as A* search
Dynamic obstacles can also be an issue


1. Did the LLM generate a working program on the first attempt?
=> Yes
2. If not, how can you improve your prompt?
=> N/A
3. What search algorithm did the LLM choose?
=> BFS
4. Why do you think the LLM selected this algorithm?
=> It is a complete search, and one which finds the shortest path.
"""


# Represents a 2D coordinate on the warehouse grid
@dataclass(frozen=True)
class Point:
    r: int
    c: int


class WarehouseAgent:
    def __init__(self, map_grid: List[str]):
        # Keep grid as a 2D list of characters for mutable visual markings
        self.grid: List[List[str]] = [list(row) for row in map_grid]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows > 0 else 0
        self.start: Optional[Point] = self._find_position('S')
        self.goal: Optional[Point] = self._find_position('G')

        # Movement vectors for Up, Down, Left, Right
        self.dr = [-1, 1, 0, 0]
        self.dc = [0, 0, -1, 1]

    # Helper to find specific characters ('S' or 'G') on the map
    def _find_position(self, target: str) -> Optional[Point]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == target:
                    return Point(r, c)
        return None  # Not found

    # Main search function using Breadth-First Search (BFS)
    def find_and_print_path(self) -> None:
        if self.start is None or self.goal is None:
            print("Error: Start (S) or Goal (G) missing from the map.")
            return

        q: deque[Point] = deque()
        # Keep track of visited nodes to prevent cycles
        visited = [[False] * self.cols for _ in range(self.rows)]
        # Keep track of parents to reconstruct the path later
        parent: List[List[Optional[Point]]] = [[None] * self.cols for _ in range(self.rows)]

        # Initialize start position
        q.append(self.start)
        visited[self.start.r][self.start.c] = True

        path_found = False

        # BFS loop
        while q:
            current = q.popleft()

            # Check if we reached the goal
            if current == self.goal:
                path_found = True
                break

            # Explore all 4 possible directions
            for i in range(4):
                next_r = current.r + self.dr[i]
                next_c = current.c + self.dc[i]

                # Ensure the next move is within bounds, not visited, and not an obstacle
                if 0 <= next_r < self.rows and 0 <= next_c < self.cols:
                    if not visited[next_r][next_c] and self.grid[next_r][next_c] != '#':
                        visited[next_r][next_c] = True
                        parent[next_r][next_c] = current  # Record where we came from
                        q.append(Point(next_r, next_c))

        if path_found:
            self._mark_and_print_path(parent)
        else:
            print("No collision-free path exists from Start to Goal.")

    # Backtracks from Goal to Start to draw the path
    def _mark_and_print_path(self, parent: List[List[Optional[Point]]]) -> None:
        assert self.goal is not None and self.start is not None
        current = parent[self.goal.r][self.goal.c]  # Start tracking back from just before the goal
        step_count = 0

        # Trace back until we hit the Start node
        while current is not None and current != self.start:
            self.grid[current.r][current.c] = '*'  # Mark the path visually
            current = parent[current.r][current.c]
            step_count += 1

        print(f"Path found! (Shortest distance: {step_count + 1} steps)")
        print("Here is the visual representation of the path:\n")

        for row in self.grid:
            print("".join(row))

    # Alias to match C++ camelCase method name
    findAndPrintPath = find_and_print_path


def main():
    # The warehouse map provided in the problem description
    warehouse_map = [
        "#####################",
        "#S....#............G#",
        "#.##....##########..#",
        "#....##.............#",
        "#.######.###.#.###..#",
        "#........#..........#",
        "#####################"
    ]

    agent = WarehouseAgent(warehouse_map)
    agent.find_and_print_path()


if __name__ == "__main__":
    main()
