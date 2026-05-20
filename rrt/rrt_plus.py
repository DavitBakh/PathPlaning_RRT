import numpy as np
from .utils import get_distance, is_collision_free
from .node import Node

class RRT:
    def __init__(self, start, goal, bounds, obstacles, step_size=1.0, goal_sample_rate=0.1, max_iter=1500, obs_type='geometric', clearance=0.5):
        self.start = Node(start)
        self.goal = Node(goal)
        self.bounds = np.array(bounds)
        self.obstacles = obstacles
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obs_type = obs_type
        self.clearance = clearance
        self.node_list = [self.start]

    def get_random_node(self):
        if np.random.rand() < self.goal_sample_rate:
            return Node(self.goal.coords)
        rand_coords = [np.random.uniform(b[0], b[1]) for b in self.bounds]
        return Node(rand_coords)

    def steer(self, from_node, to_node):
        dist = get_distance(from_node, to_node)
        if dist <= self.step_size:
            new_coords = to_node.coords
        else:
            direction = (to_node.coords - from_node.coords) / dist
            new_coords = from_node.coords + self.step_size * direction
            
        new_node = Node(new_coords)
        new_node.parent = from_node
        new_node.cost = from_node.cost + get_distance(from_node, new_node)
        return new_node

    def plan(self, draw_callback=None):
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = min(self.node_list, key=lambda n: get_distance(n, rnd_node))
            new_node = self.steer(nearest_node, rnd_node)

            if is_collision_free(nearest_node, new_node, self.obstacles, self.obs_type, self.clearance):
                self.node_list.append(new_node)
                
                if draw_callback:
                    draw_callback(nearest_node, new_node)
                
                if get_distance(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, self.goal)
                    if is_collision_free(new_node, final_node, self.obstacles, self.obs_type, self.clearance):
                        self.node_list.append(final_node)
                        if draw_callback:
                            draw_callback(new_node, final_node)
                        return self.extract_path(final_node)
        return None

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append(node.coords.tolist())
            node = node.parent
        return path[::-1]
