import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import faiss

# Load dataset
data = pd.read_csv('C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv')

# Preprocessing function
def preprocess_text(text):
    return str(text).lower().strip() if pd.notna(text) else ""

# Preprocess paper titles
data['processed_text'] = data['title'].apply(preprocess_text)

# Check if TF-IDF model exists
tfidf_model_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\models\\tfidf_vectorizer.pkl"
if os.path.exists(tfidf_model_path):
    tfidf_vectorizer = joblib.load(tfidf_model_path)
    tfidf_matrix = tfidf_vectorizer.transform(data['processed_text'])
else:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])
    joblib.dump(tfidf_vectorizer, tfidf_model_path)

# Function to evaluate KMeans model for different k values
def evaluate_kmeans_model_for_k(k_values):
    results = []

    for k in k_values:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(tfidf_matrix)

        # Calculate Silhouette Score
        labels = kmeans_model.labels_
        silhouette_avg = silhouette_score(tfidf_matrix, labels)
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

# Check if BERT embeddings exist
bert_embeddings_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\models\\bert_embeddings.npy"
if os.path.exists(bert_embeddings_path):
    bert_embeddings = np.load(bert_embeddings_path)
else:
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = bert_model.encode(data['processed_text'], convert_to_tensor=False)
    np.save(bert_embeddings_path, bert_embeddings)

# Check if FAISS index exists
faiss_index_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\models\\faiss_bert.index"
if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
else:
    dimension = bert_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(bert_embeddings).astype(np.float32))
    faiss.write_index(faiss_index, faiss_index_path)

print("✅ Models loaded or trained: TF-IDF, KMeans, BERT FAISS index.")

# **Hybrid Recommendation Function**
def recommend_papers(query, top_n=5, use_bert=True):
    query_processed = preprocess_text(query)

    # TF-IDF based KMeans
    query_vector = tfidf_vectorizer.transform([query_processed])
    
    # Ensure kmeans_model is defined
    kmeans_model = KMeans(n_clusters=5, random_state=42)
    kmeans_model.fit(tfidf_matrix)
    
    cluster_label = kmeans_model.predict(query_vector)[0]
    cluster_indices = np.where(kmeans_model.labels_ == cluster_label)[0]

    # BERT-based FAISS
    if use_bert:
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = bert_model.encode([query_processed], convert_to_tensor=False)
        D, I = faiss_index.search(np.array(query_embedding).astype(np.float32), k=top_n)
        recommended_indices = I[0]
    else:
        recommended_indices = cluster_indices[:top_n]

    # Retrieve recommended papers
    recommended_papers = []
    for idx in recommended_indices:
        recommended_papers.append({
            'id': data.iloc[idx]['id'],
            'title': data.iloc[idx]['title'],
            'authors': data.iloc[idx]['authors'],  # Include authors
            'abstract': data.iloc[idx]['abstract'],
            'citations': int(data.iloc[idx]['citations'])  # Convert np.int64 to int
        })

    return recommended_papers

# Example usage
query = "machine learning"
recommendations = recommend_papers(query, top_n=5, use_bert=True)

print("Recommended Papers:")
for paper in recommendations:
    print(f"ID: {paper['id']}, Title: {paper['title']}, Authors: {paper['authors']}, Citations: {paper['citations']}")