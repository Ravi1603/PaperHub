from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
from final_app import recommend_papers_by_keywords, build_citation_graph

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "<h1>Welcome to PaperHub API</h1>"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Log the incoming request
        print("Incoming request to /recommend")
        data = request.get_json()
        print("Request data:", data)

        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Log the query
        print("Query:", query)

        # Load data (example: loading from a CSV file)
        data = pd.read_csv("C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv")
        print("Data loaded successfully. Columns:", data.columns)

        citation_graph = build_citation_graph(data)

        # Get real recommendations
        recommendations = recommendations = recommend_papers_by_keywords(data, query, top_n=5, use_kmeans_filtering=True)
        # Convert np.int64 to regular int
        for rec in recommendations:
            rec['citations'] = int(rec['citations'])


        return jsonify({'query': query, 'recommendations': recommendations})
    except Exception as e:
        # Log the error
        print("Error in /recommend:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)