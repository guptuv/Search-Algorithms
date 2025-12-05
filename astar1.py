import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import heapq

# -----------------------------
# CONFIG
# -----------------------------
GRID_SIZE = 20
start = (5, 6)
goal = (17, 16)

# obstacles (editable)
obstacles = [
    (5,5),(5,6),(5,17),(6,5),(7,5),(16,7),(18,5),(7,7),
    (10,10),(11,10),(12,10),(10,11),(11,11),(12,11),
    (10,2),(10,11),(10,12),(11,12),(12,12),(9,14),(13,14),
    (14,7),(15,7),(16,7),(16,6),(18,16),(17,15),(16,15),
    (3,15),(4,15),(5,15),(6,15),(7,15),(8,15),(9,15),(10,15)
]

# -----------------------------
# A* FUNCTIONS
# -----------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node):
    x, y = node
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if (nx, ny) not in obstacles:
                neighbors.append((nx, ny))
    return neighbors

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    visited_sequence = []

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_sequence.append(current)

        if current == goal:
            break

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    path = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()

    return visited_sequence, path


# -----------------------------
# RUN A*
# -----------------------------
visited, path = astar(start, goal)


# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.grid(True)

# Obstacles
for (x, y) in obstacles:
    ax.scatter(x+0.5, y+0.5, c="black", s=100)

visited_scatter = ax.scatter([], [], c="skyblue", s=40)
path_scatter = ax.scatter([], [], c="red", s=60)

def update(frame):
    # 1) Show visited nodes progressively
    if frame < len(visited):
        vx = [p[0]+0.5 for p in visited[:frame]]
        vy = [p[1]+0.5 for p in visited[:frame]]
        visited_scatter.set_offsets(np.c_[vx, vy])
    else:
        visited_scatter.set_offsets(np.c_[visited])

    # 2) Show final path ONLY in the last 10 frames
    if frame > frames - 10:  # last few frames only
        px = [p[0]+0.5 for p in path]
        py = [p[1]+0.5 for p in path]
        path_scatter.set_offsets(np.c_[px, py])
    else:
        # hide final path till end
        path_scatter.set_offsets(np.empty((0, 2)))

    return visited_scatter, path_scatter



# Animation
frames = len(visited) + 40
ani = FuncAnimation(fig, update, frames=frames, interval=60, repeat=False)

plt.title("A* Path Planning Visualization")
plt.show()

# -----------------------------
# SAVE VIDEO
# -----------------------------
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save("astar_output.mp4", writer=writer)

print("Video saved as astar_output.mp4")
