from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import pandas as pd
import os
from collections import defaultdict

def preprocess_data(df):
    def preprocess_text(text):
        return str(text).lower().strip() if pd.notna(text) else ""

    df['processed_text'] = df['title'].apply(preprocess_text) + ' ' + df['abstract'].apply(preprocess_text)
   
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
   
    # Create data directory if it doesn't exist
    data_dir = os.path.join("C:\\Users\\rroganna\\Desktop\\PaperHub\\data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save models to the specified data folder
    joblib.dump(tfidf_vectorizer, os.path.join(data_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(tfidf_matrix, os.path.join(data_dir, "tfidf_matrix.pkl"))
   
    return tfidf_vectorizer, tfidf_matrix

def build_citation_graph(data):
    citation_graph = defaultdict(int)
    for _, row in data.iterrows():
        paper_id = row['id']
        citations = row.get('citations', 0)
        citation_graph[paper_id] = citations
    return citation_graph

def citation_similarity(citation_graph, paper_id1, paper_id2):
    citations1 = citation_graph.get(paper_id1, 0)
    citations2 = citation_graph.get(paper_id2, 0)
   
    if citations1 == 0 and citations2 == 0:
        return 0
   
    return min(citations1, citations2) / max(citations1, citations2)
