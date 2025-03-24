import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import os
from collections import defaultdict
from model import build_citation_graph, citation_similarity

def hybrid_recommend(keywords, top_n=5):
    data_dir = os.path.join("C:\\Users\\rroganna\\Desktop\\PaperHub\\data")
    
    tfidf_vectorizer = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
    tfidf_matrix = joblib.load(os.path.join(data_dir, "tfidf_matrix.pkl"))
    
    
    data = pd.read_csv(os.path.join(data_dir, 'arxiv_papers.csv'))
   
    keyword_vector = tfidf_vectorizer.transform([keywords])
    content_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
   
    citation_graph = build_citation_graph(data)
   
    hybrid_scores = {}
    for i, paper in enumerate(data['id']):
        citation_score = citation_similarity(citation_graph, paper, paper)
        hybrid_scores[paper] = 0.8 * content_scores[i] + 0.2 * citation_score
   
    recommended_papers = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
   
    recommendations = []
    for paper_id, score in recommended_papers:
        paper_details = data[data['id'] == paper_id].iloc[0]
        recommendations.append({
            'id': paper_details['id'],
            'title': paper_details['title'],
            'abstract': paper_details['abstract'],
            'citations': paper_details['citations']
        })
   
    return recommendations
