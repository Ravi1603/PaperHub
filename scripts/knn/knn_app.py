import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.knn.knn_model import recommend_papers, knn_model, tfidf_vectorizer, data, evaluate_knn_model_for_k
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances, precision_score, recall_score
import numpy as np

# Function to evaluate KNN model
def evaluate_knn_model():
    # Calculate pairwise distances between all samples
    distances = pairwise_distances(tfidf_vectorizer.transform(data['processed_text']), metric='cosine')
    
    # Get the indices of the nearest neighbors for each sample
    knn_indices = knn_model.kneighbors(tfidf_vectorizer.transform(data['processed_text']), n_neighbors=2, return_distance=False)
    
    # Calculate average distance to nearest neighbors
    avg_distance = np.mean([distances[i, knn_indices[i]].mean() for i in range(len(data))])
    print(f"Average distance to nearest neighbors: {avg_distance:.4f}")

    # Use the actual true labels from the dataset
    true_labels = data['category']
    print("True Labels:", true_labels)

    predicted_labels = []
    for i in range(len(data)):
        neighbors = knn_indices[i]
        neighbor_labels = data.iloc[neighbors]['category']
        predicted_label = neighbor_labels.mode()[0]  # Majority voting
        predicted_labels.append(predicted_label)

    # Calculate overall accuracy, F1 score, precision, and recall
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"Overall F1 Score: {f1:.2f}")
    print(f"Overall Precision: {precision:.2f}")
    print(f"Overall Recall: {recall:.2f}")

# Evaluate KNN model for different k values
k_values = range(1, 21)
results = evaluate_knn_model_for_k(k_values)
for result in results:
    print(f"k={result['k']}: Accuracy={result['accuracy']:.2f}, F1 Score={result['f1_score']:.2f}, Precision={result['precision']:.2f}, Recall={result['recall']:.2f}")

# Evaluate KNN model
evaluate_knn_model()

# Run the query and print recommendations
query = "machine learning"
recommendations = recommend_papers(query)

# Print recommendations in a readable format
print("\nRecommended Papers:")
for i, paper in enumerate(recommendations, 1):
    print(f"Paper {i}:")
    print(f"ID: {paper['id']}")
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract'][:300]}...")  # Truncate abstract for readability
    print(f"Citations: {paper['citations']}")
    print(f"Category: {paper['category']}")
    print("\n")