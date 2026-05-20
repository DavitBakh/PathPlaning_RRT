import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from matplotlib.widgets import Button
from rrt.rrt_plus import RRTPlus

def draw_ellipse(ax, start, goal, c_max, c_min):
    center = (np.array(start) + np.array(goal)) / 2.0
    angle = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    width = c_max
    height = math.sqrt(abs(c_max**2 - c_min**2))
    ellipse = patches.Ellipse(xy=center, width=width, height=height, angle=angle, 
                              edgecolor='purple', fc='None', lw=2, linestyle='--', alpha=0.5)
    ax.add_patch(ellipse)
    return ellipse


def run_static_2d():
    print("\n--- Ultimate RRT: Static 2D ---")
    start = [2, 2]; goal = [18, 18]
    bounds = [[0, 20], [0, 20]]
    obstacles = [(6, 6, 3), (14, 14, 3), (6, 14, 2), (14, 6, 2), (10, 10, 2.5)]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("Bidirectional Informed Adaptive RRT*-Connect")

    for (ox, oy, orad) in obstacles:
        ax.add_artist(plt.Circle((ox, oy), orad, color='gray', alpha=0.5))
        
    ax.plot(start[0], start[1], "xb", markersize=10)
    ax.plot(goal[0], goal[1], "xr", markersize=10)

    step_counter = [0]
    drawn_ellipses = []

    def draw_step(coords1, coords2, is_start_tree, is_connection=False, ellipse_cost=None):
        step_counter[0] += 1
        if is_connection:
            ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], "-m", linewidth=3)
            if ellipse_cost is not None:
                for e in drawn_ellipses: e.remove()
                drawn_ellipses.clear()
                c_min = np.linalg.norm(np.array(start) - np.array(goal))
                e = draw_ellipse(ax, start, goal, ellipse_cost, c_min)
                drawn_ellipses.append(e)
                ax.set_title(f"Path Found! Refining...")
            plt.pause(0.1)
        else:
            color = "-b" if is_start_tree else "-C1"
            ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], color, alpha=0.3)
            
        if step_counter[0] % 15 == 0:
            fig.canvas.draw_idle()
            plt.pause(0.001)

    rrt = RRTPlus(start, goal, bounds, obstacles, min_step=0.2, max_step=3.0, max_iter=800)
    path, smoothed_path, cost = rrt.plan(draw_callback=draw_step)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-g", linewidth=2, linestyle='--', label="Best RRT Path")
        
        smoothed_path = np.array(smoothed_path)
        ax.plot(smoothed_path[:, 0], smoothed_path[:, 1], color="gold", linewidth=4, label="Smoothed Path")
        
        ax.legend()
        ax.set_title(f"Finished! Final Smoothed Cost: {cost:.2f}")

    plt.ioff()
    plt.show()


def run_static_3d():
    print("\n--- Ultimate RRT: Static 3D (Optimized) ---")
    start = [2, 2, 2]; goal = [18, 18, 18]
    bounds = [[0, 20], [0, 20], [0, 20]]
    obstacles = [(10, 10, 10, 4), (5, 15, 5, 3), (15, 5, 15, 3)]

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1]); ax.set_zlim(bounds[2])
    ax.set_title("3D Bidirectional RRT*-Connect (Fast Render)")

    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 15); v = np.linspace(0, np.pi, 15)
        x = obs[0] + obs[3] * np.outer(np.cos(u), np.sin(v))
        y = obs[1] + obs[3] * np.outer(np.sin(u), np.sin(v))
        z = obs[2] + obs[3] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    ax.scatter(*start, c='b', marker='x', s=100)
    ax.scatter(*goal, c='r', marker='x', s=100)

    tree_a_line, = ax.plot([], [], [], "-b", alpha=0.3)
    tree_b_line, = ax.plot([], [], [], "-C1", alpha=0.3)
    xa, ya, za = [], [], []
    xb, yb, zb = [], [], []
    step_counter = [0]
    
    def draw_step(coords1, coords2, is_start_tree, is_connection=False, ellipse_cost=None):
        step_counter[0] += 1
        
        if is_connection:
            ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], [coords1[2], coords2[2]], "-m", linewidth=3)
            ax.set_title(f"Connection Made! Refining...")
            return

        if is_start_tree:
            xa.extend([coords1[0], coords2[0], np.nan])
            ya.extend([coords1[1], coords2[1], np.nan])
            za.extend([coords1[2], coords2[2], np.nan])
        else:
            xb.extend([coords1[0], coords2[0], np.nan])
            yb.extend([coords1[1], coords2[1], np.nan])
            zb.extend([coords1[2], coords2[2], np.nan])
            
        if step_counter[0] % 25 == 0: 
            tree_a_line.set_data_3d(xa, ya, za)
            tree_b_line.set_data_3d(xb, yb, zb)
            fig.canvas.draw_idle()
            plt.pause(0.001)

    rrt = RRTPlus(start, goal, bounds, obstacles, min_step=0.5, max_step=4.0, max_iter=800, search_radius=6.0)
    path, smoothed_path, cost = rrt.plan(draw_callback=draw_step)

    tree_a_line.set_data_3d(xa, ya, za)
    tree_b_line.set_data_3d(xb, yb, zb)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-g", linewidth=2, linestyle='--', label="Best RRT Path")
        
        smoothed_path = np.array(smoothed_path)
        ax.plot(smoothed_path[:, 0], smoothed_path[:, 1], smoothed_path[:, 2], color="gold", linewidth=4, label="Smoothed Path")
        
        ax.legend()
        ax.set_title(f"Finished! Final Smoothed Cost: {cost:.2f}")
        
    plt.ioff()
    plt.show()

def run_interactive_2d():
    print("\n--- Ultimate RRT: Interactive 2D ---")
    bounds = [[0, 20], [0, 20]]
    obstacles = [(5, 5, 2), (10, 10, 3), (15, 5, 2), (5, 15, 2), (14, 15, 2.5)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("CLICK 2 POINTS: Start (Blue), then Goal (Red)")

    for (ox, oy, orad) in obstacles:
        ax.add_artist(plt.Circle((ox, oy), orad, color='gray', alpha=0.5))
    plt.draw()

    points = plt.ginput(2, timeout=-1)
    if len(points) < 2: return

    start = list(points[0])
    goal = list(points[1])
    ax.plot(start[0], start[1], "xb", markersize=10)
    ax.plot(goal[0], goal[1], "xr", markersize=10)
    ax.set_title("Calculating Ultimate RRT...")

    plt.ion()
    drawn_ellipses = []
    def draw_step(coords1, coords2, is_start_tree, is_connection=False, ellipse_cost=None):
        if is_connection:
            ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], "-m", linewidth=3)
            if ellipse_cost is not None:
                for e in drawn_ellipses: e.remove()
                drawn_ellipses.clear()
                c_min = np.linalg.norm(np.array(start) - np.array(goal))
                e = draw_ellipse(ax, start, goal, ellipse_cost, c_min)
                drawn_ellipses.append(e)
            plt.pause(0.1)
        else:
            color = "-b" if is_start_tree else "-C1"
            ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], color, alpha=0.3)
            plt.pause(0.001)

    rrt = RRTPlus(start, goal, bounds, obstacles, min_step=0.2, max_step=3.0, max_iter=500)
    path, smoothed_path, cost = rrt.plan(draw_callback=draw_step)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-g", linewidth=2, linestyle='--', label="Path")
        
        smoothed_path = np.array(smoothed_path)
        ax.plot(smoothed_path[:, 0], smoothed_path[:, 1], color="gold", linewidth=4, label="Smoothed Path")
        
        ax.legend()
        ax.set_title(f"Finished! Final Smoothed Cost: {cost:.2f}")
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
    print("\n--- Ultimate RRT: Custom Draw 2D ---")
    bounds = [[0, 20], [0, 20]]
    start = [2, 2]; goal = [18, 18]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title("Draw obstacles (Left Click). Click RUN when done.")

    ax.plot(start[0], start[1], "xb", markersize=10)
    ax.plot(goal[0], goal[1], "xr", markersize=10)
    
    drawer = ObstacleDrawer(ax)
    ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_btn, 'RUN ULTIMATE RRT')

    def run_algorithm(event):
        drawer.disconnect()
        btn.active = False
        ax.set_title("Calculating Ultimate RRT...")
        plt.ion()
        
        drawn_ellipses = []
        step_counter = [0]
        def draw_step(coords1, coords2, is_start_tree, is_connection=False, ellipse_cost=None):
            step_counter[0] += 1
            if is_connection:
                ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], "-m", linewidth=3)
                if ellipse_cost is not None:
                    for e in drawn_ellipses: e.remove()
                    drawn_ellipses.clear()
                    c_min = np.linalg.norm(np.array(start) - np.array(goal))
                    e = draw_ellipse(ax, start, goal, ellipse_cost, c_min)
                    drawn_ellipses.append(e)
                    ax.set_title("Refining Path...")
                plt.pause(0.1)
            else:
                color = "-b" if is_start_tree else "-C1"
                ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], color, alpha=0.3)
                
            if step_counter[0] % 15 == 0:
                fig.canvas.draw_idle()
                plt.pause(0.001)

        rrt = RRTPlus(start, goal, bounds, drawer.drawn_points, 
                          min_step=0.2, max_step=2.5, max_iter=800, 
                          obs_type='drawn', clearance=0.8)
        path, smoothed_path, cost = rrt.plan(draw_callback=draw_step)

        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], "-g", linewidth=2, linestyle='--', label="Best RRT Path")
            
            smoothed_path = np.array(smoothed_path)
            ax.plot(smoothed_path[:, 0], smoothed_path[:, 1], color="gold", linewidth=4, label="Smoothed Path")
            
            ax.legend()
            ax.set_title(f"Finished! Final Smoothed Cost: {cost:.2f}")
        plt.ioff()
        plt.draw()

    btn.on_clicked(run_algorithm)
    plt.show()

if __name__ == '__main__':
    while True:
        print("\n===========================================")
        print(" ULTIMATE RRT ALGORITHM SUITE (ALL-IN-ONE)")
        print(" (Bidirectional, Informed, Adaptive, Smoothed)")
        print("===========================================")
        print("1. Static 2D Example")
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
