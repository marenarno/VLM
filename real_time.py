import os
import cv2
import torch
import numpy as np

from model_clip import load_clip_model, caption_probas
from model_gpt4 import load_gpt4_model, get_instruction
#from text_box import add_text_box

# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

# Load gpt-4 model
client, model_text = load_gpt4_model()

# # Open webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Captions categories
captions_posture = ["sitting", "standing"]
captions_cuff = ["A person holding a device", "A person holding nothing"]

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0

# Read webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Load every frame from webcam
    temp_file = 'temp_frame.jpg'
    cv2.imwrite(temp_file, frame)

    # Determine posture from probabilites
    probs = caption_probas(model_clip, preprocess_clip, temp_file, captions_posture, device)
    posture = "sitting" if np.round(probs,2)[0][0] > np.round(probs,2)[0][1] else "standing"

    # Detect device
    probs_cuff = caption_probas(model_clip, preprocess_clip, temp_file, captions_cuff, device)
    cuff = "holding the cuff" if np.round(probs_cuff,2)[0][0] > np.round(probs_cuff,2)[0][1] else "not holding the cuff"

    # Create caption
    input_text = f"Caption: The person is {posture} and {cuff}."
    print(input_text)

    # Generate instruction for every 20th frame
    if frame_count % 20 == 0:
        response = get_instruction(client, model_text, input_text, max_tokens=30)
        print(f"Assistant: {response}\n")

    # Display the frame
    cv2.imshow('Webcam', frame)
    frame_count += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 30ms or until 'q' is pressed
        break

cv2.destroyAllWindows()

