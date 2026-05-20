import numpy as np

def get_distance(node1, node2):
    return np.linalg.norm(node1.coords - node2.coords)

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    if np.linalg.norm(ab) == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def is_collision_free(node1, node2, obstacles, obs_type='geometric', clearance=0.5):
    if len(obstacles) == 0:
        return True
        
    if obs_type == 'geometric':
        dist = get_distance(node1, node2)
        if dist == 0: return True
        steps = int(dist / 0.2) + 1
        for i in range(steps + 1):
            t = i / steps
            p = node1.coords + t * (node2.coords - node1.coords)
            for obs in obstacles:
                center = np.array(obs[:-1])
                radius = obs[-1]
                if np.linalg.norm(p - center) <= radius:
                    return False
        return True
        
    elif obs_type == 'drawn':
        n1 = node1.coords
        n2 = node2.coords
        for p in obstacles:
            dist = point_to_segment_dist(p, n1, n2)
            if dist < clearance:
                return False
        return True
