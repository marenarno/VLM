import numpy as np
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torch
import clip

# Load precomputed data
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('cluster_labels.pkl', 'rb') as f:
    clusters = pickle.load(f)
embeddings = np.load('embeddings.npy')

# Load dataset
dataset_path = '/home/maren/public_demos/alpha-omega/dataset/dataset_JGCJM.csv'
dataset = pd.read_csv(dataset_path)

# Setup nearest neighbors for each cluster
cluster_indexes = {i: NearestNeighbors(n_neighbors=1, algorithm='ball_tree') for i in range(kmeans.n_clusters)}
for i in range(kmeans.n_clusters):
    cluster_embeddings = embeddings[clusters == i]
    cluster_indexes[i].fit(cluster_embeddings)

# Load model and preprocess utilities
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def get_image_embedding(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy().flatten().astype('float64')

def find_similar_image_caption(image_path):
    query_embedding = get_image_embedding(image_path, model, preprocess, device)
    query_cluster = kmeans.predict([query_embedding])[0]
    _, indices = cluster_indexes[query_cluster].kneighbors([query_embedding])
    closest_image_index = np.where(clusters == query_cluster)[0][indices[0][0]]
    return dataset.iloc[closest_image_index]['caption']

# Example usage
input_img_path = '../frames/tubing_55s/frame_0120.jpg'
print(find_similar_image_caption(input_img_path))

print("RAG system with clustering is ready to use.")

