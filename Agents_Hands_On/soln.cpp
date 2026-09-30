#include <bits/stdc++.h>
using namespace std;
/*
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
*/

// Represents a 2D coordinate on the warehouse grid
struct Point {
    int r, c;
    
    // Helper operators to compare points
    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
};

class WarehouseAgent {
private:
    vector<string> grid;
    int rows;
    int cols;
    Point start;
    Point goal;

    // Movement vectors for Up, Down, Left, Right
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    // Helper to find specific characters ('S' or 'G') on the map
    Point findPosition(char target) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == target) {
                    return {r, c};
                }
            }
        }
        return {-1, -1}; // Not found
    }

public:
    // Constructor initializes the grid and locates Start and Goal
    WarehouseAgent(const vector<string>& mapGrid) {
        grid = mapGrid;
        rows = grid.size();
        cols = grid[0].size();
        start = findPosition('S');
        goal = findPosition('G');
    }

    // Main search function using Breadth-First Search (BFS)
    void findAndPrintPath() {
        if (start.r == -1 || goal.r == -1) {
            cout << "Error: Start (S) or Goal (G) missing from the map.\n";
            return;
        }

        queue<Point> q;
        // Keep track of visited nodes to prevent cycles
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));
        // Keep track of parents to reconstruct the path later
        vector<vector<Point>> parent(rows, vector<Point>(cols, {-1, -1}));

        // Initialize start position
        q.push(start);
        visited[start.r][start.c] = true;

        bool pathFound = false;

        // BFS loop
        while (!q.empty()) {
            Point current = q.front();
            q.pop();

            // Check if we reached the goal
            if (current == goal) {
                pathFound = true;
                break;
            }

            // Explore all 4 possible directions
            for (int i = 0; i < 4; ++i) {
                int nextR = current.r + dr[i];
                int nextC = current.c + dc[i];

                // Ensure the next move is within bounds, not visited, and not an obstacle
                if (nextR >= 0 && nextR < rows && nextC >= 0 && nextC < cols) {
                    if (!visited[nextR][nextC] && grid[nextR][nextC] != '#') {
                        visited[nextR][nextC] = true;
                        parent[nextR][nextC] = current; // Record where we came from
                        q.push({nextR, nextC});
                    }
                }
            }
        }

        if (pathFound) {
            markAndPrintPath(parent);
        } else {
            cout << "No collision-free path exists from Start to Goal.\n";
        }
    }

private:
    // Backtracks from Goal to Start to draw the path
    void markAndPrintPath(const vector<vector<Point>>& parent) {
        Point current = parent[goal.r][goal.c]; // Start tracking back from just before the goal
        int stepCount = 0;
        
        // Trace back until we hit the Start node
        while (current != start) {
            grid[current.r][current.c] = '*'; // Mark the path visually
            current = parent[current.r][current.c];
            stepCount++;
        }

        cout << "Path found! (Shortest distance: " << stepCount + 1 << " steps)\n";
        cout << "Here is the visual representation of the path:\n\n";
        
        for (const string& row : grid) {
            cout << row << "\n";
        }
    }
};

int main() {
    // The warehouse map provided in the problem description
    vector<string> warehouseMap = {
        "#####################",
        "#S....#............G#",
        "#.##....##########..#",
        "#....##.............#",
        "#.######.###.#.###..#",
        "#........#..........#",
        "#####################"
    };

    WarehouseAgent agent(warehouseMap);
    agent.findAndPrintPath();

    return 0;
}