import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
from sklearn.metrics import pairwise_distances

# Suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def calculate_intracluster_distance(features, labels):
    unique_labels = np.unique(labels)
    intracluster_distances = []
    
    for label in unique_labels:
        cluster_points = features[labels == label]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            intracluster_distances.append(np.mean(distances))
    
    return np.mean(intracluster_distances)

def calculate_intercluster_distance(features, labels):
    unique_labels = np.unique(labels)
    intercluster_distances = []
    
    for i, label_i in enumerate(unique_labels):
        cluster_i_points = features[labels == label_i]
        for j, label_j in enumerate(unique_labels):
            if i != j:
                cluster_j_points = features[labels == label_j]
                distances = pairwise_distances(cluster_i_points, cluster_j_points)
                intercluster_distances.append(np.mean(distances))
    
    return np.mean(intercluster_distances)

def cluster_papers_by_category(data, n_clusters):
    # Extract features for clustering (e.g., BERT embeddings for titles, abstracts, categories)
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    #title_embeddings = bert_model.encode(data['title'], convert_to_tensor=False)
    category_embeddings = bert_model.encode(data['category'], convert_to_tensor=False)
    
    combined_features = np.array(category_embeddings)
    
    # Check for consistent number of samples
    if combined_features.shape[0] != data.shape[0]:
        raise ValueError("Number of samples in combined_features and data do not match.")
    
    # Split data into training and testing sets
    train_features, test_features, train_data, test_data = train_test_split(combined_features, data, test_size=0.2, random_state=42)
    
    # Apply normal K-means clustering on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_features)
    
    # Predict clusters for training and testing data
    train_data['cluster'] = kmeans.predict(train_features)
    test_data['cluster'] = kmeans.predict(test_features)
    
    # Evaluate clustering performance on training data
    silhouette_avg_train = silhouette_score(train_features, train_data['cluster'])
    davies_bouldin_avg_train = davies_bouldin_score(train_features, train_data['cluster'])
    calinski_harabasz_avg_train = calinski_harabasz_score(train_features, train_data['cluster'])
    intracluster_distance_train = calculate_intracluster_distance(train_features, train_data['cluster'])
    intercluster_distance_train = calculate_intercluster_distance(train_features, train_data['cluster'])
    
    print(f"Training Data - Silhouette Score: {silhouette_avg_train}")
    print(f"Training Data - Davies-Bouldin Index: {davies_bouldin_avg_train}")
    print(f"Training Data - Calinski-Harabasz Score: {calinski_harabasz_avg_train}")
    print(f"Training Data - Intracluster Distance: {intracluster_distance_train}")
    print(f"Training Data - Intercluster Distance: {intercluster_distance_train}")
    
    # Evaluate clustering performance on testing data
    silhouette_avg_test = silhouette_score(test_features, test_data['cluster'])
    davies_bouldin_avg_test = davies_bouldin_score(test_features, test_data['cluster'])
    calinski_harabasz_avg_test = calinski_harabasz_score(test_features, test_data['cluster'])
    intracluster_distance_test = calculate_intracluster_distance(test_features, test_data['cluster'])
    intercluster_distance_test = calculate_intercluster_distance(test_features, test_data['cluster'])
    
    print(f"Testing Data - Silhouette Score: {silhouette_avg_test}")
    print(f"Testing Data - Davies-Bouldin Index: {davies_bouldin_avg_test}")
    print(f"Testing Data - Calinski-Harabasz Score: {calinski_harabasz_avg_test}")
    print(f"Testing Data - Intracluster Distance: {intracluster_distance_test}")
    print(f"Testing Data - Intercluster Distance: {intercluster_distance_test}")
    
    return kmeans, silhouette_avg_test

# Example usage
data = pd.read_csv("C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv")

best_k = None
best_score = float('-inf')
best_model = None

# Try different values of k and print scores for each k
for k in range(100, 150):
    print(f"\nClustering with k={k}")
    model, score = cluster_papers_by_category(data, n_clusters=k)
    
    if score > best_score:
        best_k = k
        best_score = score
        best_model = model

# Save the model with the best score
model_path = 'best_kmeans_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

print(f"\nBest model saved with k={best_k} and Silhouette Score={best_score}")
