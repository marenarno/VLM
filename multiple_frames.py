import os
import cv2
import torch
import numpy as np
import time
import csv

from model_clip import load_clip_model, caption_probas, decide_caption#, save_probabilities
from model_gpt4 import load_gpt4_model, get_instruction
#from text_box import add_text_box

# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

# Load gpt-4 model
client, model_text = load_gpt4_model()

# Load frames
#frames_dir = '../dataset/frames/maren'
frames_dir = '../frames/tubing_55s'
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
captions_posture = ["sitting", "standing"]
captions_cuff = ["A person holding a device", "A person holding nothing"]
captions_cuff_arm = ["A person with a cuff on the arm", "A person holding a device"]
captions_cuff_placement = ["cuff on lower arm", "cuff on elbow", "cuff on upper arm"]

save_probs = True
all_probs_posture = []
all_probs_cuff = []
all_probs_cuff_placement = []
all_probs_tubing = []

# Process each frame
for i, frame_file in enumerate(frame_files):

    # Load image
    frame = cv2.imread(frame_file)

    #start_time=time.time()

    '''Determine posture from probabilites'''
    # probs_posture = caption_probas(model_clip, preprocess_clip, frame_file, captions_posture, device)
    # posture = "sitting" if np.round(probs_posture,2)[0] > np.round(probs_posture,2)[1] else "standing"
    # all_probs_posture.append(probs_posture)

    #end_time = time.time()
    #print(end_time-start_time)

    '''Detect device'''
    # probs_cuff = caption_probas(model_clip, preprocess_clip, frame_file, captions_cuff, device)
    # cuff = "holding the cuff" if np.round(probs_cuff,2)[0] > np.round(probs_cuff,2)[1] else "not holding the cuff"
    # all_probs_cuff.append(probs_cuff)

    '''Determine cuff placement'''
    captions_cuff_placement = ["device on lower arm", "device on elbow", "device on upper arm"]
    output_string_cuff_placement = captions_cuff_placement
    probs_cuff_placement  = caption_probas(model_clip, preprocess_clip, frame_file, captions_cuff_placement, device)
    cuff_placement, max_index_cuff_placement = decide_caption(output_string_cuff_placement, probs_cuff_placement)
    all_probs_cuff_placement.append(probs_cuff_placement)
    # print(f'Caption for cuff placement: {cuff_placement}')
    # print(f"Probability for '{captions_cuff_placement[0]}' is: {np.round(probs_cuff_placement,2)[0]}")
    # print(f"Probability for '{captions_cuff_placement[1]}' is: {np.round(probs_cuff_placement,2)[1]}")
    # print(f"Probability for '{captions_cuff_placement[2]}' is: {np.round(probs_cuff_placement,2)[2]}")

    '''Determine direction of tubing'''
    '''Determine direction of tubing'''
    captions_tubing = ["The grey wire on the device is going from the device pointing towards hand", 
                    "The grey wire is going from the device pointing towards the shoulder"]
    output_string_tubing = ["tubing facing down", "tubing facing up"]
    probs_tubing = caption_probas(model_clip, preprocess_clip, frame_file, captions_tubing, device)
    tubing, max_index_tubing = decide_caption(output_string_tubing, probs_tubing)
    all_probs_tubing.append(probs_tubing)
    # print(f'Caption for tubing: {tubing}')
    # print(f"Probability for '{tubing[0]}' is: {np.round(probs_tubing,2)[0]}")
    # print(f"Probability for '{tubing[1]}' is: {np.round(probs_tubing,2)[1]}")

    '''Create caption'''
    # input_text = f"Caption: The person is {posture} and {cuff}."
    # print(input_text)

    '''Generate instruction for every 20th frame'''
    # if i % 20 == 0:
    #     response = get_instruction(client, model_text, input_text, max_tokens=30)
    #     print(f"Assistant: {response}\n")

    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 30ms or until 'q' is pressed
        break

cv2.destroyAllWindows()

def save_probabilities_csv(data, filename):
    path = f'../prob_arrays/{filename}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
            print("Writing row:", row)  # Print each row to confirm what is being written

if save_probs:
        #save_probabilities(all_probs_posture, "posture_sit_device_71s")
        #save_probabilities(all_probs_cuff, "cuff_sit_device_71s")
        #save_probabilities_csv(all_probs_cuff_placement, "cuff_placement_tubing_55s_2")
        save_probabilities_csv(all_probs_tubing, "tubing_tubing_55_2s")