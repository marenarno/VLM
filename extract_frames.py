import cv2
import os

# Path to the video file
folder_name = "Maren"
video_path = f"../dataset/frames/{folder_name}.mov"

# Set up the video capture
cap = cv2.VideoCapture(video_path)

# Check if the video is opened correctly
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Create a directory to save the frames
os.makedirs(f"{folder_name}", exist_ok=True)

# Desired frame rate (e.g., extract one frame every 10 frames)
frame_skip = 5
frame_count = 0
saved_frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Finished extracting frames or error in reading frames.")
        break

    # Save every 'frame_skip' frame
    if frame_count % frame_skip == 0:
        frame_filename = f"{folder_name}/frame_{saved_frame_count:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        saved_frame_count += 1

    frame_count += 1

# When everything done, release the capture
cap.release()

