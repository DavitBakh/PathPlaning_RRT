import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from rrt.rrt_plus import RRTPlus


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

    rrt = RRTPlus(start, goal, bounds, obstacles, step_size=1.0, max_iter=1000)
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

    rrt = RRTPlus(start, goal, bounds, obstacles, step_size=2.0, max_iter=800)
    path = rrt.plan(draw_callback=draw_step)

    tree_line.set_data_3d(xs, ys, zs)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-r", linewidth=3, label="Path")
        ax.legend()
        
    plt.ioff()
    plt.show()


def run_interactive_2d():
    print("\n--- Running Interactive 2D RRT ---")
    bounds = [[0, 20], [0, 20]]
    obstacles = [(5, 5, 2), (10, 10, 3), (15, 5, 2), (5, 15, 2), (14, 15, 2.5)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("CLICK 2 POINTS: Start (Blue), then Goal (Red)")

    for (ox, oy, orad) in obstacles:
        ax.add_artist(plt.Circle((ox, oy), orad, color='gray', alpha=0.5))
    plt.draw()

    points = plt.ginput(2, timeout=-1)
    if len(points) < 2: return

    start = list(points[0])
    goal = list(points[1])
    ax.plot(start[0], start[1], "xb", markersize=10, label="Start")
    ax.plot(goal[0], goal[1], "xr", markersize=10, label="Goal")
    ax.set_title("Calculating...")

    plt.ion()
    def draw_step(from_node, to_node):
        ax.plot([from_node.coords[0], to_node.coords[0]], 
                [from_node.coords[1], to_node.coords[1]], "-g", alpha=0.4)
        plt.pause(0.001)

    rrt = RRTPlus(start, goal, bounds, obstacles, step_size=1.0, max_iter=1000)
    path = rrt.plan(draw_callback=draw_step)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-r", linewidth=2, label="Path")
        ax.legend()
    plt.ioff()
    plt.show()


class ObstacleDrawer:
    def __init__(self, ax):
        self.ax = ax
        self.drawing = False
        self.drawn_points = []
        self.c1 = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.c2 = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.c3 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.drawing = True
            self.add_point(event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 1: self.drawing = False

    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax:
            self.add_point(event.xdata, event.ydata)

    def add_point(self, x, y):
        self.drawn_points.append(np.array([x, y]))
        self.ax.add_artist(plt.Circle((x, y), 0.5, color='gray'))
        self.ax.figure.canvas.draw_idle()
        
    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.c1)
        self.ax.figure.canvas.mpl_disconnect(self.c2)
        self.ax.figure.canvas.mpl_disconnect(self.c3)

def run_custom_draw_2d():
    print("\n--- Running Custom Draw 2D RRT ---")
    bounds = [[0, 20], [0, 20]]
    start = [2, 2]; goal = [18, 18]

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("Draw obstacles (Left Click). Click RUN when done.")

    ax.plot(start[0], start[1], "xb", markersize=10, label="Start")
    ax.plot(goal[0], goal[1], "xr", markersize=10, label="Goal")
    
    drawer = ObstacleDrawer(ax)
    ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_btn, 'RUN RRT')

    def run_algorithm(event):
        drawer.disconnect()
        btn.active = False
        ax.set_title("Calculating Path...")
        plt.ion()
        
        def draw_step(from_node, to_node):
            ax.plot([from_node.coords[0], to_node.coords[0]], 
                    [from_node.coords[1], to_node.coords[1]], "-g", alpha=0.4)
            plt.pause(0.001)

        rrt = RRTPlus(start, goal, bounds, drawer.drawn_points, step_size=0.8, obs_type='drawn', clearance=0.6)
        path = rrt.plan(draw_callback=draw_step)

        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], "-r", linewidth=2, label="Path")
            ax.legend()
        plt.ioff()
        plt.draw()

    btn.on_clicked(run_algorithm)
    plt.show()


if __name__ == '__main__':
    while True:
        print("\n==================================")
        print(" RRT / RRT* ALGORITHM COLLECTION")
        print("==================================")
        print("1. Static 2D Example (RRT*)")
        print("2. Static 3D Example (Fast Rendering)")
        print("3. Interactive 2D (Click Start/Goal)")
        print("4. Custom Draw 2D (Draw Obstacles)")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ")
        
        if choice == '1': run_static_2d()
        elif choice == '2': run_static_3d()
        elif choice == '3': run_interactive_2d()
        elif choice == '4': run_custom_draw_2d()
        elif choice == '5': 
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")
