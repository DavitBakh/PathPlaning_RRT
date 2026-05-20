import matplotlib.pyplot as plt
import numpy as np
from rrt.rrt_plus import RRT


def run_static_2d():
    print("\n--- Running Static 2D RRT* ---")
    start = [2, 2]
    goal = [18, 18]
    bounds = [[0, 20], [0, 20]]
    obstacles = [(5, 5, 2), (10, 10, 3), (15, 5, 2), (5, 15, 2), (14, 15, 2.5)]

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("Static 2D RRT*")

    for (ox, oy, orad) in obstacles:
        ax.add_artist(plt.Circle((ox, oy), orad, color='gray', alpha=0.5))
    ax.plot(start[0], start[1], "xb", markersize=10, label="Start")
    ax.plot(goal[0], goal[1], "xr", markersize=10, label="Goal")

    def draw_step(from_node, to_node):
        ax.plot([from_node.coords[0], to_node.coords[0]], 
                [from_node.coords[1], to_node.coords[1]], "-g", alpha=0.4)
        plt.pause(0.001)

    rrt = RRT(start, goal, bounds, obstacles, step_size=1.0, max_iter=1000)
    path = rrt.plan(draw_callback=draw_step)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-r", linewidth=2, label="Path")
        ax.legend()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_static_2d()