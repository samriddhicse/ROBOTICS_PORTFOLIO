#!/usr/bin/env python3
"""
MAZE SOLVER - A* Pathfinding Algorithm
A robot (blue dot) finds shortest path from Start (S) to Goal (G)
while avoiding obstacles (black walls).

This is the foundation for all robot navigation systems.
"""

# ============================================================
# IMPORT STATEMENTS - Bringing in tools we need
# ============================================================

import numpy as np          # For 2D grid (like a chessboard)
import matplotlib.pyplot as plt    # For drawing the maze
import matplotlib.animation as animation  # For making the robot move
from heapq import heappush, heappop   # For priority queue (A* needs this)
import time                 # To measure how fast our algorithm runs

# ============================================================
# CLASS DEFINITION - Blueprint for our Maze Solver
# ============================================================

class MazeSolver:
    """
    A class that represents our maze solving robot.
    
    Think of a class as a blueprint. 
    'self' means "this specific maze" - like saying "my maze".
    """
    
    def __init__(self, grid_size=10, obstacle_density=0.2):
        """
        CONSTRUCTOR - This runs when we create a new maze.
        'grid_size' = how many rows and columns (10x10 by default)
        'obstacle_density' = how many walls (20% by default)
        
        Example: 
            maze = MazeSolver()  # Creates a 10x10 maze
            maze = MazeSolver(15, 0.3)  # Creates a 15x15 maze with 30% walls
        """
        
        # Store the size of our grid (e.g., 10 means 10 rows and 10 columns)
        self.size = grid_size
        
        # Create a 2D grid filled with zeros (0 = empty cell)
        # np.zeros creates an array of size grid_size x grid_size
        # Example: grid_size=3 creates:
        # [[0, 0, 0],
        #  [0, 0, 0],
        #  [0, 0, 0]]
        self.grid = np.zeros((grid_size, grid_size))
        
        # Add random obstacles (1 = wall / obstacle)
        # We loop through every cell in the grid
        for i in range(grid_size):        # i = row number (0 to grid_size-1)
            for j in range(grid_size):    # j = column number (0 to grid_size-1)
                # np.random.random() gives a random number between 0 and 1
                # If it's less than obstacle_density, we make it a wall
                if np.random.random() < obstacle_density:
                    self.grid[i][j] = 1   # 1 means wall
        
        # Set Start position (always top-left corner)
        # In a grid, (0,0) means row 0, column 0
        self.start = (0, 0)
        
        # Set Goal position (always bottom-right corner)
        # (grid_size-1, grid_size-1) means last row, last column
        self.goal = (grid_size-1, grid_size-1)
        
        # Make sure Start and Goal are empty (not walls)
        # Just in case random obstacles covered them
        self.grid[self.start] = 0    # (0,0) becomes empty
        self.grid[self.goal] = 0     # (last,last) becomes empty
        
        # These will store the path and visited cells later
        self.path = []       # Will hold the final shortest path
        self.visited = []    # Will hold all cells the algorithm explored
    
    # ============================================================
    # HEURISTIC FUNCTION - Guessing distance to goal
    # ============================================================
    
    def heuristic(self, a, b):
        """
        Heuristic: A smart guess of how far we are from the goal.
        We use Manhattan Distance: |x1-x2| + |y1-y2|
        
        Why Manhattan? Because the robot can only move up/down/left/right
        (like driving on city streets, not flying in a straight line).
        
        Example:
            Current position = (2, 3)
            Goal position = (5, 7)
            Manhattan distance = |2-5| + |3-7| = 3 + 4 = 7
        
        This is an "admissible" heuristic - it never overestimates
        the actual distance, which is required for A* to find the
        truly shortest path.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # ============================================================
    # GET NEIGHBORS - Find valid cells around a position
    # ============================================================
    
    def get_neighbors(self, pos):
        """
        Find all valid neighboring cells around a position.
        The robot can move in 4 directions: up, down, left, right.
        
        Valid means:
        1. Inside the grid (not off the board)
        2. Not a wall (not blocked)
        
        pos = (row, column) e.g., (2, 3)
        
        Returns: List of (row, column) tuples that the robot can move to.
        
        Example:
            get_neighbors((2, 3))
            Checks: up (1,3), down (3,3), left (2,2), right (2,4)
            Returns only the ones that are valid.
        """
        neighbors = []
        row, col = pos  # Unpack position into row and column
        
        # Define the 4 directions: (row_change, column_change)
        # Up: (-1,0), Down: (1,0), Left: (0,-1), Right: (0,1)
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        # Check each direction
        for dr, dc in directions:
            nr = row + dr   # New row = current row + row change
            nc = col + dc   # New column = current column + column change
            
            # Check if new position is inside the grid
            # 0 <= nr < self.size means row is valid
            # 0 <= nc < self.size means column is valid
            if 0 <= nr < self.size and 0 <= nc < self.size:
                # Check if the cell is NOT a wall (0 = empty, 1 = wall)
                if self.grid[nr][nc] == 0:
                    neighbors.append((nr, nc))  # Add to valid neighbors
        
        return neighbors
    
    # ============================================================
    # A* ALGORITHM - The core logic for finding the shortest path
    # ============================================================
    
    def solve(self):
        """
        A* (A-star) algorithm - finds the shortest path from start to goal.
        
        HOW A* WORKS:
        -----------------
        Imagine you're in a maze and want to find the exit.
        You don't know the way, but you can see the exit in the distance.
        
        A* combines two strategies:
        1. Dijkstra's algorithm: "Explore the closest cells first"
        2. Greedy Best-First: "Explore cells closest to the goal first"
        
        A* = Dijkstra + Heuristic
        
        It uses a PRIORITY QUEUE (like a to-do list) where:
        - The most promising cell is always at the front
        - "Most promising" = lowest f_score = g_score + heuristic
        
        g_score = actual cost from start to this cell
        heuristic = estimated cost from this cell to goal
        f_score = g_score + heuristic (total estimated cost)
        
        KEY TERMS:
        ----------
        - g_score: Cost from START to current cell (how far we've traveled)
        - heuristic: Estimated cost from current cell to GOAL (how far we think we need to go)
        - f_score = g_score + heuristic (total estimated journey)
        
        The algorithm explores cells with lowest f_score first,
        because those are the most promising paths.
        """
        
        # Start the timer to measure performance
        start_time = time.time()
        
        # ============================================================
        # PRIORITY QUEUE - The "to-do list" of cells to explore
        # ============================================================
        
        # Format: (f_score, counter, position)
        # Counter ensures we don't compare positions (just tie-breaking)
        open_set = []
        counter = 0
        
        # Put the start position in the queue with f_score = heuristic(start, goal)
        heappush(open_set, (0, counter, self.start))
        
        # ============================================================
        # TRACKING VARIABLES - Keep track of what we've learned
        # ============================================================
        
        # came_from = "Where did I come from?"
        # Like leaving breadcrumbs to trace the path back
        # Example: came_from[(2,3)] = (1,3) means we came from (1,3) to get to (2,3)
        came_from = {}
        
        # g_score = "How far did I travel from start?"
        # Actual cost from start to each cell
        # Dictionary: key = position, value = cost
        g_score = {self.start: 0}
        
        # f_score = "Total estimated journey cost"
        # g_score + heuristic (actual cost + estimated remaining cost)
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        
        # visited = set of cells we've already explored
        # Set is faster than list for checking "is this cell already visited?"
        visited = set()
        
        # ============================================================
        # MAIN LOOP - Explore until we find the goal or run out of cells
        # ============================================================
        
        while open_set:
            # Pop the cell with lowest f_score from the priority queue
            # heappop removes and returns the smallest item
            current = heappop(open_set)[2]  # [0]=f_score, [1]=counter, [2]=position
            
            # Mark current cell as visited
            visited.add(current)
            self.visited.append(current)  # Store for visualization
            
            # ============================================================
            # CHECK: Did we reach the goal?
            # ============================================================
            
            if current == self.goal:
                # Reconstruct the path by following the breadcrumbs backwards
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                path.reverse()  # Reverse to get start -> goal order
                self.path = path
                
                # Print success information
                print(f"✓ Path found in {len(path)} steps!")
                print(f"✓ Explored {len(visited)} cells")
                print(f"✓ Time: {(time.time() - start_time)*1000:.2f}ms")
                return True
            
            # ============================================================
            # EXPLORE NEIGHBORS - Look at all valid cells around current
            # ============================================================
            
            for neighbor in self.get_neighbors(current):
                # Calculate tentative g_score (cost from start to this neighbor)
                # Cost = g_score[current] + 1 (each move costs 1)
                tentative_g = g_score[current] + 1
                
                # If this path to neighbor is better than any previous path:
                # Either we haven't visited neighbor yet, or we found a cheaper path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update the breadcrumb - we came from 'current'
                    came_from[neighbor] = current
                    
                    # Update g_score with the better cost
                    g_score[neighbor] = tentative_g
                    
                    # Update f_score = g_score + heuristic
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, self.goal)
                    
                    # Add neighbor to the priority queue for future exploration
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        # If we exit the loop without finding the goal, no path exists
        print("✗ No path found!")
        return False
    
    # ============================================================
    # VISUALIZATION - Make the algorithm come to life
    # ============================================================
    
    def visualize(self):
        """
        Animate the maze solver in action!
        
        Shows:
        1. The maze with walls (black)
        2. Start (green S) and Goal (red G)
        3. Explored cells (light blue, fading)
        4. The robot (dark blue dot) following the final path
        
        This uses Matplotlib's FuncAnimation to create frames.
        """
        
        # Create a figure and axis for drawing
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            """
            This function is called for each frame of the animation.
            'frame' = which frame number we're on.
            """
            
            # Clear the previous frame
            ax.clear()
            
            # ============================================================
            # DRAW: Walls (black rectangles)
            # ============================================================
            
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i][j] == 1:
                        # plt.Rectangle((x, y), width, height, color)
                        # Note: matplotlib uses (x, y) where:
                        # x = column, y = row (but y is flipped)
                        # self.size-1-i flips the row so row 0 is at the top
                        ax.add_patch(plt.Rectangle(
                            (j, self.size-1-i), 1, 1, color='black'
                        ))
            
            # ============================================================
            # DRAW: Explored cells (light blue, fading)
            # ============================================================
            
            for idx, pos in enumerate(self.visited[:frame]):
                if pos not in [self.start, self.goal]:
                    # Fading effect: earlier cells are lighter
                    if len(self.visited) > 0:
                        alpha = 0.3 + 0.5 * (idx / len(self.visited))
                    else:
                        alpha = 0.3
                    ax.add_patch(plt.Rectangle(
                        (pos[1], self.size-1-pos[0]), 1, 1,
                        color='lightblue', alpha=alpha
                    ))
            
            # ============================================================
            # DRAW: Walls AGAIN (so they appear on top of explored cells)
            # ============================================================
            
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i][j] == 1:
                        ax.add_patch(plt.Rectangle(
                            (j, self.size-1-i), 1, 1, color='black'
                        ))
            
            # ============================================================
            # DRAW: Start (green) and Goal (red)
            # ============================================================
            
            # Start: Green rectangle with 'S'
            ax.add_patch(plt.Rectangle(
                (self.start[1], self.size-1-self.start[0]), 1, 1,
                color='green', alpha=0.7
            ))
            ax.text(self.start[1]+0.3, self.size-1-self.start[0]+0.3, 
                    'S', fontsize=14, fontweight='bold')
            
            # Goal: Red rectangle with 'G'
            ax.add_patch(plt.Rectangle(
                (self.goal[1], self.size-1-self.goal[0]), 1, 1,
                color='red', alpha=0.7
            ))
            ax.text(self.goal[1]+0.3, self.size-1-self.goal[0]+0.3,
                    'G', fontsize=14, fontweight='bold')
            
            # ============================================================
            # DRAW: Path and Robot
            # ============================================================
            
            if self.path and frame >= len(self.visited):
                # After exploration, show the robot following the path
                path_frame = frame - len(self.visited)
                
                # Draw the path cells (blue)
                for idx in range(min(path_frame, len(self.path))):
                    pos = self.path[idx]
                    if pos not in [self.start, self.goal]:
                        ax.add_patch(plt.Rectangle(
                            (pos[1], self.size-1-pos[0]), 1, 1,
                            color='blue', alpha=0.6
                        ))
                
                # Draw the robot (dark blue dot)
                if path_frame < len(self.path):
                    pos = self.path[path_frame]
                    ax.add_patch(plt.Circle(
                        (pos[1]+0.5, self.size-1-pos[0]+0.5), 0.3,
                        color='darkblue', zorder=10
                    ))
            
            # ============================================================
            # SETUP: Grid boundaries and labels
            # ============================================================
            
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_xticks([])  # Hide x-axis numbers
            ax.set_yticks([])  # Hide y-axis numbers
            ax.set_aspect('equal')  # Make cells square
            ax.set_title('Maze Solver - A* Pathfinding', fontsize=14)
        
        # ============================================================
        # CREATE ANIMATION
        # ============================================================
        
        # Total frames = exploration frames + path following frames
        total_frames = len(self.visited) + len(self.path)
        
        # FuncAnimation creates the animation
        # fig = the figure to draw on
        # update = function to call for each frame
        # frames = total number of frames
        # interval = time between frames (milliseconds)
        ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                      interval=100, repeat=False)
        
        # Display the animation
        plt.show()
        return ani

# ============================================================
# MAIN EXECUTION - This runs when we execute the file
# ============================================================

if __name__ == "__main__":
    """
    This is the entry point of our program.
    'if __name__ == "__main__"' means:
    "Only run this code if this file is executed directly,
     not if it's imported as a module."
    
    Think of it as: "This is where the program starts."
    """
    
    # Print colorful welcome message
    print("🤖 MAZE SOLVER - A* Pathfinding Algorithm")
    print("=" * 50)
    print("🧠 Skills you'll learn:")
    print("   - Python: Classes, Lists, Dictionaries, Sets")
    print("   - Algorithms: A* Pathfinding, Heuristics")
    print("   - Math: Manhattan Distance, Grid Navigation")
    print("   - Visualization: Matplotlib Animations")
    print("   - Linux: Terminal commands, File management")
    print("   - Git: Version control, Commit, Push")
    print("=" * 50)
    
    # ============================================================
    # CREATE MAZE
    # ============================================================
    
    # You can change these values to create different mazes:
    # grid_size=10 means 10x10 grid
    # obstacle_density=0.2 means 20% of cells are walls
    solver = MazeSolver(grid_size=10, obstacle_density=0.2)
    
    # Print maze information
    print(f"\n📊 Maze Information:")
    print(f"   Grid Size: {solver.size}x{solver.size}")
    print(f"   Start: {solver.start}")
    print(f"   Goal: {solver.goal}")
    print(f"   Obstacles: {int(np.sum(solver.grid))} cells")
    print(f"   Empty Cells: {solver.size*solver.size - int(np.sum(solver.grid))}")
    
    # ============================================================
    # SOLVE MAZE
    # ============================================================
    
    print("\n🔍 Finding shortest path...")
    print("   (A* is exploring the maze, trying different paths)")
    
    if solver.solve():
        print(f"\n📍 Final Path: {len(solver.path)} steps")
        print(f"   Path: {solver.path[:5]}...{solver.path[-5:]}")
        print("\n🎬 Animation starting...")
        solver.visualize()
    else:
        print("\n❌ No path found! Try running again for a different maze.")
