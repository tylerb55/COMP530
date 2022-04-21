import numpy as np
from collections import deque
import cv2


"""preprocess image frames to remove unimportant information"""#can crop frame when you know how much to crop
def preprocess_frame(frame):
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    normalised_frame=grey/255.0
    preprocessed_frame=cv2.resize(normalised_frame,(110,84))
    #preprocessed_frame=normalised_frame.reshape(110,84)
    return preprocessed_frame
    
"""stack the frames"""
stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((110,84), dtype=np.int32) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int32) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        stacked_state=np.reshape(stacked_state, [1,110,84,4])
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
        stacked_state=np.reshape(stacked_state, [1,110,84,4])
    
    return stacked_state, stacked_frames