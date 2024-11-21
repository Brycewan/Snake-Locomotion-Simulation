import os
import numpy as np
import pygame  # pygame for visualization
import cv2  # for video creation
import mesh
import time_integrator
import parameters

pygame.init()

# parameters
tol = parameters.TOLERANCE
length = parameters.LENGTH
width = parameters.WIDTH
height = parameters.HEIGHT
n = parameters.N_SEG
rho = parameters.RHO
initial_stretch = parameters.INITIAL_STRETCH

k_value = parameters.K
wave_speed = parameters.WAVE_SPEED
amplitude = parameters.AMPLITUDE
wave_length = parameters.WAVE_LENGTH
h = parameters.TIME_STEP  # simulation time step

###########################################################################

# initializations
[x, e, seg_index] = mesh.generate_3d_snake(length, width, height, n)

v = np.array([[0.0, 0.0, 0.0]] * len(x))             # velocity
m = [rho * length * width * height / n] * len(x)  # calculate node mass evenly
# rest length squared
l2 = []
for i in range(0, len(e)):
    diff = x[e[i][0]] - x[e[i][1]]
    l2.append(diff.dot(diff))
# l0 = np.sqrt(l2)    # rest length   
k = [k_value] * len(e)    # spring stiffness

###########################################################################

# Frame saving settings
frames_dir = f"demos/{k_value}_{wave_speed}_{amplitude}"
os.makedirs(frames_dir, exist_ok=True)
fps = 30  # frames per second for the final video
steps_per_frame = int(1 / (fps * h))  # how many time steps per frame

# simulation with visualization
resolution = np.array([1200, 800])
offset = resolution / 2
scale = 150

def screen_projection(x_local):
    """Project a point in simulation space to screen space."""
    return [offset[0] + scale * x_local[0], resolution[1] - (offset[1] + scale * x_local[1])]

def draw_grid(screen, snake_head, grid_spacing=0.5, grid_color=(200, 200, 200)):
    """Draw a background grid that moves relative to the snake."""
    global_offset = np.array(snake_head) % grid_spacing  # Offset due to snake movement
    for x_line in np.arange(-10, 10, grid_spacing):
        x_line_adjusted = x_line - global_offset[0]
        start_pos = screen_projection([x_line_adjusted, -10, 0])
        end_pos = screen_projection([x_line_adjusted, 10, 0])
        pygame.draw.aaline(screen, grid_color, start_pos, end_pos)
    for y_line in np.arange(-10, 10, grid_spacing):
        y_line_adjusted = y_line - global_offset[1]
        start_pos = screen_projection([-10, y_line_adjusted, 0])
        end_pos = screen_projection([10, y_line_adjusted, 0])
        pygame.draw.aaline(screen, grid_color, start_pos, end_pos)
        
# def screen_projection(x):
#     return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

###########################################################################

# Simulation loop
time_step = 0
screen = pygame.display.set_mode(resolution)
running = True

while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    print('### Time step', time_step, '###')
    
    # Compute snake head position
    snake_head = (x[int(n/2*4)] + x[int(n/2*4)+1] + x[int(n/2*4)+2] + x[int(n/2*4)+3]) / 4 # Assume the first node is the snake's head
    x_centered = x - snake_head  # Shift all positions to center the head
    
    # fill the background and draw the square
    screen.fill((255, 255, 255))
    draw_grid(screen, snake_head)
    # Draw the snake in its centered coordinates
    for eI in e:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x_centered[eI[0]]), screen_projection(x_centered[eI[1]]))
    for xI in x_centered:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * length * scale)
    # for eI in e:
    #     pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    # for xI in x:
    #     pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * length * scale)
    pygame.display.flip()   # flip the display
    
    # Save frames at specified intervals
    if time_step % steps_per_frame == 0:
        frame_filename = os.path.join(frames_dir, f"frame_{time_step:05d}.png")
        pygame.image.save(screen, frame_filename)
        print(f"Saved frame {frame_filename}")
    
    [x, v] = time_integrator.step_forward(x, e, v, m, l2, k, h, tol, time_step * h, seg_index, wave_speed, amplitude, wave_length)
    time_step += 1
    
    # if time_step > 1200:  # Stop after a large number of steps, or define a specific condition
    #     running = False
    
pygame.quit()

# Video creation
video_filename = "simulation_video.mp4"
video_filename = os.path.join(frames_dir, video_filename)
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
if frame_files:
    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as {video_filename}")
else:
    print("No frames found to create video.")
    
    