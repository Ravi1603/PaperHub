import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import faiss

# Load dataset
data = pd.read_csv('C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\arxiv_papers.csv')

# Preprocessing function
def preprocess_text(text):
    return str(text).lower().strip() if pd.notna(text) else ""

# Preprocess paper titles
data['processed_text'] = data['title'].apply(preprocess_text)

# Check if TF-IDF model exists
tfidf_model_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\tfidf_vectorizer.pkl"
if os.path.exists(tfidf_model_path):
    tfidf_vectorizer = joblib.load(tfidf_model_path)
    tfidf_matrix = tfidf_vectorizer.transform(data['processed_text'])
else:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])
    joblib.dump(tfidf_vectorizer, tfidf_model_path)

# Check if KNN model exists
knn_model_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\knn_model.pkl"
if os.path.exists(knn_model_path):
    knn_model = joblib.load(knn_model_path)
else:
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_model.fit(tfidf_matrix)
    joblib.dump(knn_model, knn_model_path)

# Check if BERT embeddings exist
bert_embeddings_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\bert_embeddings.npy"
if os.path.exists(bert_embeddings_path):
    bert_embeddings = np.load(bert_embeddings_path)
else:
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = bert_model.encode(data['processed_text'], convert_to_tensor=False)
    np.save(bert_embeddings_path, bert_embeddings)

# Check if FAISS index exists
faiss_index_path = "C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\faiss_bert.index"
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
            'abstract': data.iloc[idx]['abstract'],
            'citations': data.iloc[idx]['citations']
        })

    return recommended_papers