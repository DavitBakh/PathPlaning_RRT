import numpy as np
import math

def get_distance(node1, node2):
    return np.linalg.norm(node1.coords - node2.coords)

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    if np.linalg.norm(ab) == 0: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_nearest_obstacle_distance(coords, obstacles, obs_type='geometric'):
    if not obstacles or len(obstacles) == 0:
        return float('inf')
        
    if obs_type == 'geometric':
        min_dist = float('inf')
        for obs in obstacles:
            center = np.array(obs[:-1])
            radius = obs[-1]
            dist = np.linalg.norm(coords - center) - radius
            if dist < min_dist:
                min_dist = dist
        return min_dist
        
    elif obs_type == 'drawn':
        obstacles_arr = np.array(obstacles)
        dists = np.linalg.norm(obstacles_arr - coords, axis=1)
        return np.min(dists)

def is_collision_free(coords1, coords2, obstacles, obs_type='geometric', clearance=0.5):
    if len(obstacles) == 0:
        return True
        
    if obs_type == 'geometric':
        dist = np.linalg.norm(coords1 - coords2)
        if dist == 0: return True
        steps = int(dist / 0.1) + 1
        for i in range(steps + 1):
            t = i / steps
            p = coords1 + t * (coords2 - coords1)
            for obs in obstacles:
                center = np.array(obs[:-1])
                radius = obs[-1]
                if np.linalg.norm(p - center) <= radius:
                    return False
        return True
        
    elif obs_type == 'drawn':
        for p in obstacles:
            if point_to_segment_dist(p, coords1, coords2) < clearance:
                return False
        return True

def sample_ellipse(start_coords, goal_coords, c_max, c_min, bounds):
    if len(start_coords) == 3:
        return np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        
    center = (start_coords + goal_coords) / 2.0
    angle = math.atan2(goal_coords[1] - start_coords[1], goal_coords[0] - start_coords[0])
    r1 = c_max / 2.0
    r2 = math.sqrt(abs(c_max**2 - c_min**2)) / 2.0
    
    rho = math.sqrt(np.random.random())
    theta = np.random.uniform(0, 2*math.pi)
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    
    x_rot = x * r1 * math.cos(angle) - y * r2 * math.sin(angle)
    y_rot = x * r1 * math.sin(angle) + y * r2 * math.cos(angle)
    
    return np.array([center[0] + x_rot, center[1] + y_rot])
