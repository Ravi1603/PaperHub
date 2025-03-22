from scripts.load_embeddings import load_embeddings
from scripts.hybrid_recommend import hybrid_recommend
import pandas as pd

# Load dataset
data = pd.read_csv('data/combined_papers.csv')

# Load embeddings and models
tfidf_matrix, tfidf_vectorizer, index = load_embeddings()

# Example usage of hybrid_recommend function
user_keywords = "Adversarial attacks"
recommended_papers = hybrid_recommend(user_keywords, tfidf_vectorizer, tfidf_matrix, data)

# Display the recommended papers in a readable format
for i, paper in enumerate(recommended_papers, 1):
    print(f"Paper {i}:")
    print(f"ID: {paper['id']}")
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract']}")
    print(f"Citations: {paper['citations']}")
    print("\n")