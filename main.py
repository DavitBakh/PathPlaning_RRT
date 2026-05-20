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


def run_static_3d():
    print("\n--- Running Static 3D RRT* (Optimized) ---")
    start = [2, 2, 2]
    goal = [18, 18, 18]
    bounds = [[0, 20], [0, 20], [0, 20]]
    obstacles = [(10, 10, 10, 4), (5, 15, 5, 3), (15, 5, 15, 3)]

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1]); ax.set_zlim(bounds[2])
    ax.set_title("Static 3D RRT* (Fast Rendering)")

    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 15); v = np.linspace(0, np.pi, 15)
        x = obs[0] + obs[3] * np.outer(np.cos(u), np.sin(v))
        y = obs[1] + obs[3] * np.outer(np.sin(u), np.sin(v))
        z = obs[2] + obs[3] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    ax.scatter(*start, c='b', marker='x', s=100, label="Start")
    ax.scatter(*goal, c='r', marker='x', s=100, label="Goal")

    tree_line, = ax.plot([], [], [], "-g", alpha=0.3)
    xs, ys, zs = [], [], []
    step_counter = [0]
    
    def draw_step(from_node, to_node):
        step_counter[0] += 1
        
        xs.extend([from_node.coords[0], to_node.coords[0], np.nan])
        ys.extend([from_node.coords[1], to_node.coords[1], np.nan])
        zs.extend([from_node.coords[2], to_node.coords[2], np.nan])
        
        if step_counter[0] % 25 == 0: 
            tree_line.set_data_3d(xs, ys, zs)
            fig.canvas.draw_idle()
            plt.pause(0.001)

    rrt = RRT(start, goal, bounds, obstacles, step_size=2.0, max_iter=800)
    path = rrt.plan(draw_callback=draw_step)

    tree_line.set_data_3d(xs, ys, zs)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-r", linewidth=3, label="Path")
        ax.legend()
        
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    #run_static_2d()
    run_static_3d()