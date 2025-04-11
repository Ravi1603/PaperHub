import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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

# Check if KNN model exists
knn_model_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\models\\knn_model.pkl"
if os.path.exists(knn_model_path):
    knn_model = joblib.load(knn_model_path)
else:
    knn_model = NearestNeighbors(n_neighbors=2, metric='cosine')
    knn_model.fit(tfidf_matrix)
    joblib.dump(knn_model, knn_model_path)

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

print("✅ Models loaded or trained: TF-IDF, KNN, BERT FAISS index.")

# **Hybrid Recommendation Function**
def recommend_papers(query, top_n=5, use_bert=True):
    query_processed = preprocess_text(query)

    # TF-IDF based KNN
    query_vector = tfidf_vectorizer.transform([query_processed])
    _, knn_indices = knn_model.kneighbors(query_vector, n_neighbors=top_n)

    # BERT-based FAISS
    if use_bert:
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = bert_model.encode([query_processed], convert_to_tensor=False)
        D, I = faiss_index.search(np.array(query_embedding).astype(np.float32), k=top_n)
        recommended_indices = I[0]
    else:
        recommended_indices = knn_indices[0]

    # Retrieve recommended papers
    recommended_papers = []
    for idx in recommended_indices:
        recommended_papers.append({
            'id': data.iloc[idx]['id'],
            'title': data.iloc[idx]['title'],
            'authors': data.iloc[idx]['authors'],  # Include authors
            'abstract': data.iloc[idx]['abstract'],
            'citations': int(data.iloc[idx]['citations']),  # Convert np.int64 to int
            'category': data.iloc[idx]['category']  # Include category
        })

    return recommended_papers

# Function to evaluate KNN model for different k values
def evaluate_knn_model_for_k(k_values):
    results = []

    for k in k_values:
        knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn_model.fit(tfidf_matrix)

        # Get the indices of the nearest neighbors for each sample
        knn_indices = knn_model.kneighbors(tfidf_vectorizer.transform(data['processed_text']), n_neighbors=k, return_distance=False)

        # Use the actual true labels from the dataset
        true_labels = data['category']

        predicted_labels = []
        for i in range(len(data)):
            neighbors = knn_indices[i]
            neighbor_labels = data.iloc[neighbors]['category']
            predicted_label = neighbor_labels.mode()[0]  # Majority voting
            predicted_labels.append(predicted_label)

        # Calculate accuracy, F1 score, precision, and recall
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')

        results.append({
            'k': k,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })

    return results