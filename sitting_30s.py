import cv2
import torch
import numpy as np
import time

from model_clip import load_clip_model, caption_probas, decide_caption

# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

sitting_threshold = 0.7  # Probability threshold for detecting 'sitting'
required_sitting_time = 30  # Required sitting time in seconds

# Initialize counters
sitting_time = 0
countdown_started = False
countdown_start_time = 0

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame to a temporary file for processing
    frame_file = 'temp_frame.jpg'
    cv2.imwrite(frame_file, frame)

    # Determine posture from probabilities
    captions_posture = ["sitting", "standing"]
    probs_posture = caption_probas(model_clip, preprocess_clip, frame_file, captions_posture, device)
    posture, max_index_posture = decide_caption(captions_posture, probs_posture)

    if posture == "sitting":
        if countdown_started:
            sitting_time = time.time() - countdown_start_time
            if sitting_time >= required_sitting_time:
                print("Please apply your blood pressure cuff.")
                countdown_started = False
                sitting_time = 0
        else:
            print(f"Please relax for {required_sitting_time - int(sitting_time)} more seconds.")
    else:
        if not countdown_started:
            print("Please sit down and relax for 30 seconds before monitoring your blood pressure.")
            countdown_start_time = time.time()
            countdown_started = True

    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 30ms or until 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()