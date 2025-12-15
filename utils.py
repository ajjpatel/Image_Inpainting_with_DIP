import torch
import torch.nn.functional as F
import random
import math

def make_mask(image_size, mask_size, mask_type='mixed'):
    # Validate and adjust mask size if needed
    if mask_size >= image_size:
        mask_size = image_size - 2  # Leave at least 1 pixel margin
    
    if mask_type == 'mixed':
        mask_types = ['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches']
        mask_type = random.choice(mask_types)
    
    if mask_type == 'square':
        return make_square_mask(image_size, mask_size)
    elif mask_type == 'circle':
        return make_circle_mask(image_size, mask_size)
    elif mask_type == 'triangle':
        return make_triangle_mask(image_size, mask_size)
    elif mask_type == 'ellipse':
        return make_ellipse_mask(image_size, mask_size)
    elif mask_type == 'irregular':
        return make_irregular_mask(image_size, mask_size)
    elif mask_type == 'random_patches':
        return make_random_patches_mask(image_size, mask_size)
    else:
        return make_square_mask(image_size, mask_size)

def make_square_mask(image_size, mask_size):
    # Ensure mask_size is valid
    mask_size = min(mask_size, image_size - 2)
    top = torch.randint(0, image_size - mask_size + 1, (1,)).item()
    left = torch.randint(0, image_size - mask_size + 1, (1,)).item()
    mask = torch.zeros((1, image_size, image_size))
    mask[:, top:top+mask_size, left:left+mask_size] = 1
    return mask

def make_circle_mask(image_size, mask_size):
    # Ensure radius is valid
    radius = min(mask_size // 2, (image_size - 2) // 2)
    center_x = torch.randint(radius, image_size - radius, (1,)).item()
    center_y = torch.randint(radius, image_size - radius, (1,)).item()
    
    mask = torch.zeros((1, image_size, image_size))
    
    y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
    
    distance = torch.sqrt((x - center_x).float()**2 + (y - center_y).float()**2)
    
    mask[0] = (distance <= radius).float()
    
    return mask

def make_triangle_mask(image_size, mask_size):
    mask = torch.zeros((1, image_size, image_size))
    
    # Ensure mask_size is valid
    mask_size = min(mask_size, image_size - 2)
    margin = mask_size // 2
    center_x = torch.randint(margin, image_size - margin, (1,)).item()
    center_y = torch.randint(margin, image_size - margin, (1,)).item()
    height = int(mask_size * 0.866)
    vertices = [
        (center_x, center_y - height // 2), 
        (center_x - mask_size // 2, center_y + height // 2), 
        (center_x + mask_size // 2, center_y + height // 2)  
    ]
    
    for y in range(image_size):
        for x in range(image_size):
            if point_in_triangle((x, y), vertices):
                mask[0, y, x] = 1
    
    return mask

def make_ellipse_mask(image_size, mask_size):
    mask = torch.zeros((1, image_size, image_size))
    
    # Ensure mask_size is valid
    mask_size = min(mask_size, image_size - 2)
    margin = mask_size // 2
    center_x = torch.randint(margin, image_size - margin, (1,)).item()
    center_y = torch.randint(margin, image_size - margin, (1,)).item()
    
    a = mask_size // 2 
    b = torch.randint(mask_size // 4, mask_size // 2, (1,)).item() 
    angle = torch.rand(1).item() * 2 * math.pi 
    
    y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
    
    x_centered = x.float() - center_x
    y_centered = y.float() - center_y
    
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_rot = x_centered * cos_angle + y_centered * sin_angle
    y_rot = -x_centered * sin_angle + y_centered * cos_angle
    
    ellipse_eq = (x_rot / a)**2 + (y_rot / b)**2
    mask[0] = (ellipse_eq <= 1).float()
    
    return mask

def make_irregular_mask(image_size, mask_size):
    mask = torch.zeros((1, image_size, image_size))
    
    # Ensure mask_size is valid
    mask_size = min(mask_size, image_size - 2)
    margin = max(1, mask_size // 4)
    center_x = torch.randint(margin, image_size - margin, (1,)).item()
    center_y = torch.randint(margin, image_size - margin, (1,)).item()
    
    points = [(center_x, center_y)]
    current_x, current_y = center_x, center_y
    
    num_steps = mask_size * 3
    for _ in range(num_steps):
        dx = torch.randint(-3, 4, (1,)).item()
        dy = torch.randint(-3, 4, (1,)).item()
        
        new_x = max(0, min(image_size - 1, current_x + dx))
        new_y = max(0, min(image_size - 1, current_y + dy))
        
        points.append((new_x, new_y))
        current_x, current_y = new_x, new_y
    
    if len(points) > 2:
        hull_points = convex_hull(points)
        for y in range(image_size):
            for x in range(image_size):
                if point_in_polygon((x, y), hull_points):
                    mask[0, y, x] = 1
    
    mask = F.conv2d(mask.unsqueeze(0), 
                    gaussian_kernel(5, 1.0).unsqueeze(0).unsqueeze(0), 
                    padding=2).squeeze(0)
    mask = (mask > 0.3).float()
    
    return mask

def make_random_patches_mask(image_size, mask_size):
    mask = torch.zeros((1, image_size, image_size))
    
    # Ensure mask_size is valid
    mask_size = min(mask_size, image_size - 2)
    num_patches = torch.randint(3, 8, (1,)).item()
    patch_size = max(1, mask_size // 3)
    
    for _ in range(num_patches):
        patch_type = random.choice(['square', 'circle'])
        
        if patch_type == 'square':
            top = torch.randint(0, image_size - patch_size + 1, (1,)).item()
            left = torch.randint(0, image_size - patch_size + 1, (1,)).item()
            mask[:, top:top+patch_size, left:left+patch_size] = 1
        else:
            radius = max(1, patch_size // 2)
            if radius > 0:
                center_x = torch.randint(radius, image_size - radius, (1,)).item()
                center_y = torch.randint(radius, image_size - radius, (1,)).item()
                
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                distance = torch.sqrt((x - center_x).float()**2 + (y - center_y).float()**2)
                patch_mask = (distance <= radius).float()
                mask[0] = torch.maximum(mask[0], patch_mask)
    
    return mask

def point_in_triangle(point, triangle):
    x, y = point
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
    
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

def convex_hull(points):
    if len(points) < 3:
        return points
    
    points = sorted(set(points))
    if len(points) < 3:
        return points
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g)
