import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


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

embeddings = np.vstack(embeddings)

# Assuming `embeddings` is your array of image embeddings
embeddings = np.array(embeddings).astype('float64')  # Ensure correct data type

# Explicitly setting n_init to suppress FutureWarning
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0).fit(embeddings)


# Perform K-Means clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
clusters = kmeans.labels_

# Create Nearest Neighbor index for each cluster
cluster_indexes = {i: NearestNeighbors(n_neighbors=1, algorithm='ball_tree') for i in range(num_clusters)}
for i in range(num_clusters):
    cluster_embeddings = embeddings[clusters == i]
    if len(cluster_embeddings) > 0:
        cluster_indexes[i].fit(cluster_embeddings)

# Function to find the nearest neighbor within the same cluster
def find_similar_image_caption(query_image_path):
    query_embedding = get_image_embeddings([query_image_path], model, preprocess, device).flatten()
    query_cluster = kmeans.predict([query_embedding])[0]
    _, indices = cluster_indexes[query_cluster].kneighbors([query_embedding])
    closest_image_index = np.where(clusters == query_cluster)[0][indices[0][0]]
    return dataset.iloc[closest_image_index]['caption']

# Example usage
#input_img_path = '../frames/tubing_55s/frame_0085.jpg'
input_img_path = '../frames/tubing_55s/frame_0120.jpg'
print(find_similar_image_caption(input_img_path))

