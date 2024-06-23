import torch
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

import sys
import os

# Path to the directory containing model_clip.py
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

from model_clip import load_clip_model

# Get image embeddings
def get_image_embedding(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy().flatten()


# Load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = load_clip_model(device)

# Load dataset
dataset_path = '/home/maren/public_demos/alpha-omega/dataset/dataset_JGCJM.csv'
dataset = pd.read_csv(dataset_path)

# Process and store image embeddings for all images in dataset
embeddings = []
for index, row in dataset.iterrows():
    image_path = row['image']  # Adjust this if your column name is different
    embedding = get_image_embedding(image_path, model_clip, preprocess_clip, device)
    embeddings.append(embedding)

# Convert embeddings to a df and save to csv
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv("/dataset_image_emb.csv", index=False)
embeddings = [get_image_embedding(path, model_clip, preprocess_clip, device) for path in dataset['image']]



# Fit Nearest Neighbors
# nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
# nn_model.fit(embeddings)

# # Function to find and return the most similar image caption
# def find_similar_image_caption(query_image_path):
#     query_embedding = get_image_embedding(query_image_path, model, preprocess, device)
#     distances, indices = nn_model.kneighbors([query_embedding])
#     return dataset.iloc[indices[0][0]]['caption']

# # Example usage
# #input_img_path = '../frames/tubing_55s/frame_0085.jpg'
# input_img_path = '../frames/tubing_55s/frame_0120.jpg'
# print(find_similar_image_caption(input_img_path))
