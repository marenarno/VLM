import os
import cv2
import torch
import numpy as np


from model_clip import load_clip_model, caption_probas



# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

# Load frames from video and categories
frames_dir = './frames_97s'
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
captions = ["sitting", "standing"]


# Display frames and give feedback
frame_display_delay = 30 # Delay in ms between frame displays

for frame_file in frame_files:
    
    # Display file
    frame = cv2.imread(frame_file)
    cv2.imshow('Frame', frame)

    # Determine posture from probabilites
    probs = caption_probas(model_clip, preprocess_clip, frame_file, captions, device)
    posture = "sitting" if probs[0][0] > probs[0][1] else "sitting"



