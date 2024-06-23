import torch
import clip
from PIL import Image
import numpy as np
import pickle
import os

def load_clip_model(device):
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

'''
This function takes in a image or MORE? Can it do both? Yes but how? 

'''
def caption_probas(model, preprocess, image_filename, captions, device):
    """
    Zero-shot learning to find the probability of each caption for the image
    
    Args:
    - model and preprocess (torch.nn.Module): CLIP
    - image_filename (str): Path to the image file for which caption probs needs to be calculated
    - captions (list): List of captions to evaluate against the image
    - device (torch.device): cpu
    
    Returns:
    - numpy.ndarray: 1D numpy array containing the probs for each caption
    """

    image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
    text = clip.tokenize(captions).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
    return probs

def decide_caption(output_string, probs):
    max_index = np.argmax(np.round(probs, 2))
    return output_string[1], max_index

# def save_probabilities(probs, name_location):
#     # Ensure the directory exists
#     os.makedirs('../prob_arrays', exist_ok=True)
#     # Save the probabilities to a file
#     with open(f'../prob_arrays/{name_location}.pkl', 'wb') as f:
#         pickle.dump(probs, f)

import csv

def save_probabilities(data, filename):
    path = f'../probs_arrays/{filename}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            if not isinstance(row, (list, tuple)):
                row = [row]  # Ensure row is iterable
            writer.writerow(row)





''' Testing if the person is sitting or standing '''
# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# # Image and text
# image_filename ='../frames/frames_97s/frame_0007.jpg'
# captions = ["sitting", "standing"]
# probs = caption_probas(model, preprocess, image_filename, captions, device)
# print("")
# print(f"Image filename: {image_filename}")
# print(f"Probability for '{captions[0]}' is: {np.round(probs,2)[0][0]}")
# print(f"Probability for '{captions[1]}' is: {np.round(probs,2)[0][1]}")
# print("")

''' Testing if the person is holding the blood pressure cuff or not '''
# Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# # Load and preprocess the image
# image_filename = "../frames/holding_device_30s/frame_0048.jpg"
# captions = ["A person holding a device", "A person holding nothing"]
# probs = caption_probas(model, preprocess, image_filename, captions, device)

# print("")
# print(f"Probability for '{captions[0]}' is: {np.round(probs,2)[0][0]}")
# print(f"Probability for '{captions[1]}' is: {np.round(probs,2)[0][1]}")
# print("")



