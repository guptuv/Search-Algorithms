import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
step_size = 0.2
max_iters = 2000

start = (1.0, 1.0)
goal  = (8.0, 8.0)

obstacles = [
    (5,5), (5,6),
    (6,5), (6,6)
]

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def near_obstacle(p):
    for o in obstacles:
        if dist(p, o) < 1.0:
            return True
    return False

def steer(nearest, rnd):
    dx = rnd[0] - nearest[0]
    dy = rnd[1] - nearest[1]
    d  = math.sqrt(dx*dx + dy*dy)

    if d == 0:
        return nearest

    new_x = nearest[0] + step_size * (dx / d)
    new_y = nearest[1] + step_size * (dy / d)
    return (round(new_x, 3), round(new_y, 3))

# -------------------------------------------------
# MAIN RRT LOGIC
# -------------------------------------------------
nodes = [start]
parents = {start: None}
tree_edges = []
goal_reached = False

for it in range(1, max_iters + 1):

    rnd = (random.uniform(0, 10),
           random.uniform(0, 10))

    if near_obstacle(rnd):
        continue

    nearest = min(nodes, key=lambda n: dist(n, rnd))
    new_node = steer(nearest, rnd)

    if near_obstacle(new_node):
        continue

    nodes.append(new_node)
    parents[new_node] = nearest
    tree_edges.append((nearest, new_node))

    print(f"Iteration {it}: Added {new_node}")

    if dist(new_node, goal) < 1.0:
        parents[goal] = new_node
        nodes.append(goal)
        tree_edges.append((new_node, goal))
        print(f"Goal reached at iteration {it}!")
        goal_reached = True
        break

# -------------------------------------------------
# RECONSTRUCT PATH
# -------------------------------------------------
final_path = []
if goal_reached:
    cur = goal
    while cur is not None:
        final_path.append(cur)
        cur = parents[cur]
    final_path.reverse()

# -------------------------------------------------
# PLOTTING + ANIMATION FOR VS CODE
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

def draw_base():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True)
    # obstacles
    for o in obstacles:
        ax.plot(o[0], o[1], "ks", markersize=12)
    # start + goal
    ax.scatter(start[0], start[1], c="green", s=100)
    ax.scatter(goal[0], goal[1], c="red", s=100)

def init():
    ax.clear()
    draw_base()

def update(i):
    ax.clear()
    draw_base()

    # draw edges up to i
    for a, b in tree_edges[:i]:
        ax.plot([a[0], b[0]], [a[1], b[1]], "b-", linewidth=1)

    # final path only on last frame
    if goal_reached and i == len(tree_edges) - 1:
        px = [p[0] for p in final_path]
        py = [p[1] for p in final_path]
        ax.plot(px, py, "r-", linewidth=3)

ani = FuncAnimation(fig, update, frames=len(tree_edges), interval=50)

# Save MP4 video
writer = FFMpegWriter(fps=30)
ani.save("RRT.mp4", writer=writer)

print("\nVideo saved as RRT.mp4")

# Show final static frame
plt.show()
