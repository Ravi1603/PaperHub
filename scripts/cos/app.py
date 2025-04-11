import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.cos.load_data import load_data_from_db
from scripts.cos.model import preprocess_data
from scripts.cos.reco import hybrid_recommend

# Load data from CSV
df = load_data_from_db()

if df is not None:
    # Preprocess data
    tfidf_vectorizer, tfidf_matrix = preprocess_data(df)

    # Example usage of hybrid_recommend function
    user_keywords = "Adversarial attacks"
    recommended_papers = hybrid_recommend(user_keywords)

    # Display the recommended papers in a readable format
    for i, paper in enumerate(recommended_papers, 1):
        print(f"Paper {i}:")
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {paper['authors']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"Citations: {paper['citations']}")
        print("\n")
else:
    print("Failed to load data. Cannot continue.")
