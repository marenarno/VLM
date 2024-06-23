import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import os

# Define functions to load model, preprocess and embed images
def load_clip_model(device):
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

def get_image_embeddings(image_paths, model, preprocess, device):
    images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths]
    images = torch.cat(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
    return image_features.cpu().numpy()

# Load dataset
dataset_path = '/home/maren/public_demos/alpha-omega/dataset/dataset_JGCJM.csv'
dataset = pd.read_csv(dataset_path)

# Load model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_clip_model(device)

# Embed all images in batches
batch_size = 32
embeddings = []
for i in range(0, len(dataset['image']), batch_size):
    batch_paths = dataset['image'][i:i + batch_size]
    batch_embeddings = get_image_embeddings(batch_paths, model, preprocess, device)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings).astype('float64')  # Ensure correct data type

# Perform K-Means clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(embeddings)
clusters = kmeans.labels_

# Create Nearest Neighbor index for each cluster
cluster_indexes = {i: NearestNeighbors(n_neighbors=1, algorithm='ball_tree') for i in range(num_clusters)}
for i in range(num_clusters):
    cluster_embeddings = embeddings[clusters == i]
    cluster_indexes[i].fit(cluster_embeddings)

# Function to find the nearest neighbor within the same cluster
def find_similar_image_caption(query_image_path):
    query_embedding = get_image_embeddings([query_image_path], model, preprocess, device).flatten()
    query_cluster = kmeans.predict([query_embedding])[0]
    _, indices = cluster_indexes[query_cluster].kneighbors([query_embedding])
    closest_image_index = np.where(clusters == query_cluster)[0][indices[0][0]]
    return dataset.iloc[closest_image_index]['caption']

# Process each frame
frames_dir = '../frames/tubing_55s'
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

captions = []
for frame_file in frame_files:
    caption = find_similar_image_caption(frame_file)
    captions.append((frame_file, caption))

# Save captions to a CSV file
captions_df = pd.DataFrame(captions, columns=['ImagePath', 'Caption'])
captions_df.to_csv('image_captions.csv', index=False)

print("Captions for all images saved to 'image_captions.csv'.")
