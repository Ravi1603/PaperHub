import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv('C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv')

# Preprocess categories
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# One-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
category_vectors = onehot_encoder.fit_transform(data[['category_encoded']])

# Function to evaluate KMeans model for different k values
def evaluate_kmeans_model_for_k(k_values):
    results = []

    for k in k_values:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(category_vectors)

        # Calculate Silhouette Score
        labels = kmeans_model.labels_
        silhouette_avg = silhouette_score(category_vectors, labels)
        results.append({
            'k': k,
            'silhouette_score': silhouette_avg
        })

    return results

# Define range of k values to evaluate
k_values = range(2, 21)

# Evaluate KMeans model for different k values and print results
results = evaluate_kmeans_model_for_k(k_values)
for result in results:
    print(f"k={result['k']}: Silhouette Score={result['silhouette_score']:.4f}")

# Example usage
query = "machine learning"
query_category = "cs.AI"  # Example category for the query

# Encode query category
query_category_encoded = label_encoder.transform([query_category])
query_category_vector = onehot_encoder.transform([query_category_encoded])

# Predict cluster for query category
kmeans_model = KMeans(n_clusters=5, random_state=42)
kmeans_model.fit(category_vectors)
cluster_label = kmeans_model.predict(query_category_vector)[0]
cluster_indices = np.where(kmeans_model.labels_ == cluster_label)[0]

# Retrieve recommended papers
recommended_papers = []
for idx in cluster_indices[:5]:  # Top 5 recommendations
    recommended_papers.append({
        'id': data.iloc[idx]['id'],
        'title': data.iloc[idx]['title'],
        'authors': data.iloc[idx]['authors'],  # Include authors
        'abstract': data.iloc[idx]['abstract'],
        'citations': int(data.iloc[idx]['citations']),  # Convert np.int64 to int
        'category': data.iloc[idx]['category']  # Include category
    })

print("Recommended Papers:")
for paper in recommended_papers:
    print(f"ID: {paper['id']}, Title: {paper['title']}, Authors: {paper['authors']}, Citations: {paper['citations']}, Category: {paper['category']}")