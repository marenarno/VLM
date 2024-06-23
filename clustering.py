import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

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

# Perform K-Means clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(embeddings.astype('float64'))
clusters = kmeans.labels_

# Save the model, labels, and embeddings
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('cluster_labels.pkl', 'wb') as f:
    pickle.dump(clusters, f)
np.save('embeddings.npy', embeddings)

print("Clustering and data saving complete.")
