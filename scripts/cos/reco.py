import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from scripts.cos.model import build_citation_graph, citation_similarity

def hybrid_recommend(keywords, top_n=5):
    data_dir = os.path.join("C:\\Users\\rroganna\\Desktop\\PaperHub\\models")
    
    data = pd.read_csv("C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv")
    
    # Load or compute BERT embeddings
    bert_embeddings_path = os.path.join(data_dir, "bert_embeddings.npy")
    if os.path.exists(bert_embeddings_path):
        bert_embeddings = np.load(bert_embeddings_path)
    else:
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        bert_embeddings = bert_model.encode(data['processed_text'], convert_to_tensor=False)
        np.save(bert_embeddings_path, bert_embeddings)
    
    # Load or create FAISS index
    faiss_index_path = os.path.join(data_dir, "faiss_bert.index")
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
    else:
        dimension = bert_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(bert_embeddings).astype(np.float32))
        faiss.write_index(faiss_index, faiss_index_path)
    
    # Compute BERT similarity using FAISS
    keyword_vector = bert_model.encode([keywords], convert_to_tensor=False)
    _, indices = faiss_index.search(np.array(keyword_vector).astype(np.float32), top_n)
    
    # Build citation graph
    citation_graph = build_citation_graph(data)
    
    # Compute hybrid scores
    hybrid_scores = {}
    for idx in indices[0]:
        paper_id = data.iloc[idx]['id']
        content_score = cosine_similarity([keyword_vector], [bert_embeddings[idx]])[0][0]
        
        # Compute average citation similarity with other papers
        citation_score = np.mean([citation_similarity(citation_graph, paper_id, other_paper) for other_paper in data['id'] if other_paper != paper_id])
        
        # Combine scores with weights
        hybrid_scores[paper_id] = 0.7 * content_score + 0.3 * citation_score
    
    # Sort papers by hybrid scores in descending order
    recommended_papers = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    recommendations = []
    for paper_id, score in recommended_papers:
        paper_details = data[data['id'] == paper_id].iloc[0]
        recommendations.append({
            'id': paper_details['id'],
            'title': paper_details['title'],
            'abstract': paper_details['abstract'],
            'citations': paper_details['citations'],
            'authors': paper_details['authors']
        })
    
    return recommendations

# Example usage
keywords = "machine learning"
recommendations = hybrid_recommend(keywords)
print(recommendations)