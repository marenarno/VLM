import os
import cv2
import torch
import numpy as np
import time

from model_clip import load_clip_model, caption_probas, decide_caption
from model_gpt4 import load_gpt4_model, get_instruction
from text_box import display_text_window
#from text_box import add_text_box

# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

# Load gpt-4 model
client, model_text = load_gpt4_model()

# Load frame
frame_dir = "../dataset/frames/maren/maren_0040.jpg"

# Display file
frame = cv2.imread(frame_dir)

#start_time=time.time()
#end_time = time.time()

'''Determine posture from probabilites'''
captions_posture = ["sitting", "standing"]
probs_posture = caption_probas(model_clip, preprocess_clip, frame_dir, captions_posture, device)
posture, max_index_posture = decide_caption(captions_posture, probs_posture)
print(f'Caption for posture: {posture}')

'''Determine cuff status from probabilites'''
captions_cuff = ["A person with a device on arm", "A person holding a device","A person with no device"]
output_string_cuff = ["have the cuff on the arm", "have the cuff in hand", "is not holding the cuff"]
probs_cuff = caption_probas(model_clip, preprocess_clip, frame_dir, captions_cuff, device)
cuff, max_index_cuff = decide_caption(output_string_cuff, probs_cuff)
print(f'Caption for cuff: {cuff}')

'''Determine direction of tubing'''
captions_tubing = ["The grey wire on the device is going from the device to the hand", 
                   "The tubing from blood pressure cuff is going from the cuff and is pointing against the shoulder"]
#captions_tubing = ["The tubing on the blood cuff is in the right direction", "The "]
output_string_tubing = ["tubing facing down", "tubing facing up"]
probs_tubing = caption_probas(model_clip, preprocess_clip, frame_dir, captions_tubing, device)
tubing, max_index_tubing = decide_caption(output_string_tubing, probs_tubing)
print(f'Caption for tubing: {tubing}')
print(f"Probability for '{tubing[0]}' is: {np.round(probs_tubing,2)[0]}")
print(f"Probability for '{tubing[1]}' is: {np.round(probs_tubing,2)[1]}")

'''Determine cuff placement'''
captions_cuff_placement = ["cuff on lower arm", "cuff on elbow", "cuff on upper arm"]
output_string_cuff_placement = captions_cuff_placement
probs_cuff_placement  = caption_probas(model_clip, preprocess_clip, frame_dir, captions_cuff_placement, device)
cuff_placement, max_index_cuff_placement = decide_caption(output_string_cuff_placement, probs_cuff_placement)
print(f'Caption for cuff placement: {cuff_placement}')
print(f"Probability for '{captions_cuff_placement[0]}' is: {np.round(probs_cuff_placement,2)[0]}")
print(f"Probability for '{captions_cuff_placement[1]}' is: {np.round(probs_cuff_placement,2)[1]}")
print(f"Probability for '{captions_cuff_placement[2]}' is: {np.round(probs_cuff_placement,2)[2]}")

# Create caption
# input_text = f"Caption: The person is {posture} and the person {cuff}."
# if cuff == output_string_cuff[0]:
#     input_text += f" The {tubing}."
# print(input_text)

# response = get_instruction(client, model_text, input_text, max_tokens=30)
# print(f"Assistant: {response}\n")

# # Create the text window image
# text_window = display_text_window(frame, response)

# Show the frame and the text window side by side
# combined_img = np.hstack((frame, text_window))
# cv2.imshow('Frame and Text', combined_img)
cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()