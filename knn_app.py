from scripts.knn_model import recommend_papers, knn_model, tfidf_vectorizer, data
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances
import numpy as np

# Function to evaluate KNN model
def evaluate_knn_model():
    # Calculate pairwise distances between all samples
    distances = pairwise_distances(tfidf_vectorizer.transform(data['processed_text']), metric='cosine')
    
    # Get the indices of the nearest neighbors for each sample
    knn_indices = knn_model.kneighbors(tfidf_vectorizer.transform(data['processed_text']), n_neighbors=10, return_distance=False)
    
    # Calculate average distance to nearest neighbors
    avg_distance = np.mean([distances[i, knn_indices[i]].mean() for i in range(len(data))])
    
    print(f"Average distance to nearest neighbors: {avg_distance:.4f}")

    # For simplicity, let's assume we have true labels for evaluation
    true_labels = data['category']  # Replace with actual labels if available
    predicted_labels = []
    
    for i in range(len(data)):
        neighbors = knn_indices[i]
        neighbor_labels = data.iloc[neighbors]['category']
        predicted_label = neighbor_labels.mode()[0]  # Majority voting
        predicted_labels.append(predicted_label)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Evaluate KNN model
evaluate_knn_model()

# Run the query and print recommendations
query = "Deep Learning adversarial attacks"
recommendations = recommend_papers(query)

print("\nRecommendations:")
for paper in recommendations:
    print(paper)