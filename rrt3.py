import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
step_size = 0.1
max_iters = 20000

start = (3.0, 2.0)
goal  = (18.0, 18.0)

obstacles = [
    (5,5),(5,6),(5,17),(6,5),(7,5),(16,7),(18,5),(7,7),
    (10,10),(11,10),(12,10),(10,11),(11,11),(12,11),
    (10,2),(10,11),(10,12),(11,12),(12,12),(9,14),(13,14),
    (14,7),(15,7),(16,7),(16,6),(18,16),(17,15),(16,15),
    (3,15),(4,15),(5,15),(6,15),(7,15),(8,15),(9,15),(10,15)
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

    rnd = (random.uniform(0, 20),
           random.uniform(0, 20))

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
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
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


from matplotlib.animation import FFMpegWriter, PillowWriter
import shutil

def safe_save_animation(animation, filename_mp4="RRT-3.mp4", fps=30):
    """
    Safely save Matplotlib animation:
      - If ffmpeg is installed → saves MP4
      - Otherwise → saves GIF fallback
    """
    print("\n=== Saving Animation ===")

    # 1. Check if FFmpeg exists
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is not None:
        print(f"FFmpeg found at: {ffmpeg_path}")
        try:
            writer = FFMpegWriter(fps=fps, codec="libx264")
            animation.save(filename_mp4, writer=writer)
            print(f"✔ MP4 saved successfully as {filename_mp4}")
            return
        except Exception as e:
            print("\n⚠ FFmpeg failed to write MP4!")
            print("Error:", e)
            print("Switching to GIF instead...\n")

    else:
        print("⚠ FFmpeg NOT FOUND. Saving as GIF instead.\n")

    # 2. GIF fallback
    gif_name = filename_mp4.replace(".mp4", ".gif")
    animation.save(gif_name, writer=PillowWriter(fps=fps))
    print(f"✔ GIF saved successfully as {gif_name}")

# Save MP4 video
safe_save_animation(ani, "RRT3.mp4", fps=30)



print("\nVideo saved as RRT3.mp4")

# Show final static frame
plt.show()
