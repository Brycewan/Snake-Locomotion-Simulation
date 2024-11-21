# ANCHOR: generate
import numpy as np
import os

def generate(side_length, n_seg):
    # sample nodes uniformly on a square
    x = np.array([[0.0, 0.0]] * ((n_seg + 1) ** 2))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]
    
    # connect the nodes with edges
    e = []
    # horizontal edges
    for i in range(0, n_seg):
        for j in range(0, n_seg + 1):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j])
    # vertical edges
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    # diagonals
    for i in range(0, n_seg):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
            e.append([(i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])

    return [x, e]
# ANCHOR_END: generate

# ANCHOR: write_to_file
def write_to_file(frameNum, x, n_seg):
    # Check if 'output' directory exists; if not, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # create obj file
    filename = f"output/{frameNum}.obj"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in x:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n") 
        # write vertex indices for each triangle
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                #NOTE: each cell is exported as 2 triangles for rendering
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1}\n")
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1} {i * (n_seg+1) + j+1 + 1}\n")
# ANCHOR_END: write_to_file


def generate_snake(length, width, n_segments):
    x = np.array([[0.0, 0.0]] * 2 * (n_segments + 1))
    # head
    x[0] = np.array([[0.0, +width/2]])
    x[1] = np.array([[0.0, -width/2]])
    # body
    for i in range(0, n_segments):
        x[2 * (i + 1)] = x[2 * i] - np.array([[length, 0.0]])
        x[2 * (i + 1) + 1] = x[2 * i + 1] - np.array([[length, 0.0]])
        
    # connect the nodes with edges
    e = []
    segment_indices = []
    # horizontal edges
    for i in range(0, n_segments + 1):
        e.append([2 * i, 2 * i + 1])
        segment_indices.append(-1)
    # vertical edges
    for i in range(0, n_segments):
        e.append([2 * i, 2 * i + 2])
        segment_indices.append(i) # segment_indices[n_segments + 1 + i] = i
        e.append([2 * i + 1, 2 * i + 3])
        segment_indices.append(i) # segment_indices[n_segments + 1 + i + 1] = i
        
    # diagonals
    for i in range(0, n_segments):
        e.append([2 * i, 2 * i + 3])
        segment_indices.append(-1)
        e.append([2 * i + 1, 2 * i + 2])
        segment_indices.append(-1)
    
    return [x, e, segment_indices]

def generate_3d_snake(length, width, height, n_segments):
    # mass points
    x = np.array([[0.0, 0.0, 0.0]] * 4 * (n_segments + 1))
    # head
    x[0] = np.array([[0.0, +width/2, height/2]])
    x[1] = np.array([[0.0, -width/2, height/2]])
    x[2] = np.array([[0.0, +width/2, -height/2]])
    x[3] = np.array([[0.0, -width/2, -height/2]])
    # body
    for i in range(0, n_segments):
        x[4 * (i + 1)] = x[4 * i] - np.array([[length, 0.0, 0.0]])
        x[4 * (i + 1) + 1] = x[4 * i + 1] - np.array([[length, 0.0, 0.0]])
        x[4 * (i + 1) + 2] = x[4 * i + 2] - np.array([[length, 0.0, 0.0]])
        x[4 * (i + 1) + 3] = x[4 * i + 3] - np.array([[length, 0.0, 0.0]])
        
    # connect the nodes with edges
    e = []
    segment_indices = [] 
    # [a,b,c] 
    # a: 0 for joints, 1 for segments; 
    # b: segment index; 
    # c: 
    # joint: 0 for left, 1 for right, 2 for top, 3 for bottom, 4 for diagonals
    # segment: 0 for left top, 1 for right top, 2 for left bottom, 3 for right bottom, 4 for diagonals
    
    # segments
    for i in range(0, n_segments):
        # left top
        e.append([4 * i, 4 * (i + 1)])
        segment_indices.append([1, i, 0])
        # right top
        e.append([4 * i + 1, 4 * (i + 1) + 1])
        segment_indices.append([1, i, 1])
        # left bottom
        e.append([4 * i + 2, 4 * (i + 1) + 2])
        segment_indices.append([1, i, 2])
        # right bottom
        e.append([4 * i + 3, 4 * (i + 1) + 3])
        segment_indices.append([1, i, 3])
    
    # joints
    for i in range(0, n_segments + 1):
        # left
        e.append([4 * i, 4 * i + 2])
        segment_indices.append([0, i, 0])
        # right
        e.append([4 * i + 1, 4 * i + 3])
        segment_indices.append([0, i, 1])
        # top
        e.append([4 * i, 4 * i + 1])
        segment_indices.append([0, i, 2])
        # bottom
        e.append([4 * i + 2, 4 * i + 3])
        segment_indices.append([0, i, 3])

    # diagonals
    for i in range(0, n_segments + 1):
        e.append([4 * i, 4 * i + 3])
        segment_indices.append([-1, i, 4])
        e.append([4 * i + 1, 4 * i + 2])
        segment_indices.append([-1, i, 4])
    for i in range(0, n_segments):
        e.append([4 * i, 4 * (i + 1) + 1])
        segment_indices.append([-1, i, 4])
        e.append([4 * i, 4 * (i + 1) + 2])
        segment_indices.append([-1, i, 4])
        
        e.append([4 * i + 1, 4 * (i + 1)])
        segment_indices.append([-1, i, 4])
        e.append([4 * i + 1, 4 * (i + 1) + 3])
        segment_indices.append([-1, i, 4])
        
        e.append([4 * i + 2, 4 * (i + 1)])
        segment_indices.append([-1, i, 4])
        e.append([4 * i + 2, 4 * (i + 1) + 3])
        segment_indices.append([-1, i, 4])
        
        e.append([4 * i + 3, 4 * (i + 1) + 1])
        segment_indices.append([-1, i, 4])
        e.append([4 * i + 3, 4 * (i + 1) + 2])
        segment_indices.append([-1, i, 4])
    
    return [x, e, segment_indices]