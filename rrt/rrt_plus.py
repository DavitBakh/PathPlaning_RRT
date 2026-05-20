import numpy as np
from utils import Node, get_distance, get_nearest_obstacle_distance, is_collision_free, sample_ellipse

class RRTPlus:
    def __init__(self, start, goal, bounds, obstacles, min_step=0.2, max_step=3.0, max_iter=1000, search_radius=5.0, obs_type='geometric', clearance=0.5):
        self.start = Node(start)
        self.goal = Node(goal)
        self.bounds = np.array(bounds)
        self.obstacles = obstacles
        
        self.min_step = min_step
        self.max_step = max_step
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.obs_type = obs_type
        self.clearance = clearance
        
        self.tree_a = [self.start]
        self.tree_b = [self.goal]
        self.is_tree_a_start = True
        
        self.c_min = get_distance(self.start, self.goal)
        self.c_best = float('inf')
        self.best_path = None

    def get_random_node(self):
        if self.c_best < float('inf'):
            coords = sample_ellipse(self.start.coords, self.goal.coords, self.c_best, self.c_min, self.bounds)
            for i in range(len(self.bounds)):
                coords[i] = np.clip(coords[i], self.bounds[i][0], self.bounds[i][1])
            return Node(coords)
        else:
            rand_coords = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            return Node(rand_coords)

    def get_adaptive_step(self, coords):
        dist_obs = get_nearest_obstacle_distance(coords, self.obstacles, self.obs_type)
        if self.obs_type == 'drawn':
            dist_obs -= self.clearance
        step = max(self.min_step, min(self.max_step, dist_obs * 0.8))
        return step

    def steer(self, from_node, to_node):
        dist = get_distance(from_node, to_node)
        step_size = self.get_adaptive_step(from_node.coords)
        
        if dist <= step_size:
            new_coords = to_node.coords
        else:
            direction = (to_node.coords - from_node.coords) / dist
            new_coords = from_node.coords + step_size * direction
            
        new_node = Node(new_coords)
        new_node.parent = from_node
        new_node.cost = from_node.cost + get_distance(from_node, new_node)
        return new_node

    def get_near_nodes(self, tree, new_node):
        nnode = len(tree) + 1
        r = min(self.search_radius * (np.log(nnode) / nnode) ** 0.5, self.max_step * 3)
        return [node for node in tree if get_distance(node, new_node) <= r]

    def insert_node(self, tree, nearest_node, new_node):
        near_nodes = self.get_near_nodes(tree, new_node)
        min_cost = new_node.cost
        best_parent = nearest_node

        for near_node in near_nodes:
            if is_collision_free(near_node.coords, new_node.coords, self.obstacles, self.obs_type, self.clearance):
                cost = near_node.cost + get_distance(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_node

        new_node.parent = best_parent
        new_node.cost = min_cost
        tree.append(new_node)

        for near_node in near_nodes:
            if is_collision_free(new_node.coords, near_node.coords, self.obstacles, self.obs_type, self.clearance):
                new_cost = new_node.cost + get_distance(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    self.update_costs(tree, near_node)
        return new_node

    def update_costs(self, tree, start_node):
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            for child in tree:
                if child.parent == node:
                    child.cost = node.cost + get_distance(node, child)
                    queue.append(child)

    def extract_path(self, node_a, node_b):
        path_start, path_goal = [], []
        if self.is_tree_a_start:
            n1, n2 = node_a, node_b
        else:
            n1, n2 = node_b, node_a
            
        curr = n1
        while curr is not None:
            path_start.append(curr.coords.tolist())
            curr = curr.parent
            
        curr = n2
        while curr is not None:
            path_goal.append(curr.coords.tolist())
            curr = curr.parent
            
        return path_start[::-1] + path_goal

    def smooth_path(self, path, iterations=200):
        if path is None or len(path) <= 2:
            return path
        
        smoothed_path = path.copy()
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                break
            
            i = np.random.randint(0, len(smoothed_path) - 1)
            j = np.random.randint(i + 1, len(smoothed_path))
            
            if j - i <= 1:
                continue
                
            p1 = np.array(smoothed_path[i])
            p2 = np.array(smoothed_path[j])
            
            if is_collision_free(p1, p2, self.obstacles, self.obs_type, self.clearance):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
                
        return smoothed_path

    def plan(self, draw_callback=None):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            
            nearest_a = min(self.tree_a, key=lambda n: get_distance(n, rnd_node))
            new_node_a = self.steer(nearest_a, rnd_node)
            
            if is_collision_free(nearest_a.coords, new_node_a.coords, self.obstacles, self.obs_type, self.clearance):
                new_node_a = self.insert_node(self.tree_a, nearest_a, new_node_a)
                
                if draw_callback:
                    draw_callback(new_node_a.parent.coords, new_node_a.coords, self.is_tree_a_start)
                
                nearest_b = min(self.tree_b, key=lambda n: get_distance(n, new_node_a))
                new_node_b = self.steer(nearest_b, new_node_a)
                
                if is_collision_free(nearest_b.coords, new_node_b.coords, self.obstacles, self.obs_type, self.clearance):
                    new_node_b = self.insert_node(self.tree_b, nearest_b, new_node_b)
                    if draw_callback:
                        draw_callback(new_node_b.parent.coords, new_node_b.coords, not self.is_tree_a_start)
                    
                    if get_distance(new_node_a, new_node_b) <= self.min_step and is_collision_free(new_node_a.coords, new_node_b.coords, self.obstacles, self.obs_type, self.clearance):
                        total_cost = new_node_a.cost + new_node_b.cost + get_distance(new_node_a, new_node_b)
                        if total_cost < self.c_best:
                            self.c_best = total_cost
                            self.best_path = self.extract_path(new_node_a, new_node_b)
                            if draw_callback:
                                draw_callback(new_node_a.coords, new_node_b.coords, True, is_connection=True, ellipse_cost=self.c_best)

            self.tree_a, self.tree_b = self.tree_b, self.tree_a
            self.is_tree_a_start = not self.is_tree_a_start

        if self.best_path:
            smoothed = self.smooth_path(self.best_path)
            smoothed_cost = sum(np.linalg.norm(np.array(smoothed[k]) - np.array(smoothed[k+1])) for k in range(len(smoothed)-1))
            return self.best_path, smoothed, smoothed_cost
            
        return None, None, float('inf')
