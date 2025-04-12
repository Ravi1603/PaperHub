from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import pandas as pd
from final_app import recommend_papers_by_keywords, build_citation_graph

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="together",
    api_key=HF_API_KEY,
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "<h1>Welcome to PaperHub API</h1>"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        print("Incoming request to /recommend")
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        df = pd.read_csv("C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\all_papers.csv")
        citation_graph = build_citation_graph(df)

        recommendations = recommend_papers_by_keywords(df, query, top_n=5, use_kmeans_filtering=True)
        for rec in recommendations:
            rec['citations'] = int(rec['citations'])

        return jsonify({'query': query, 'recommendations': recommendations})
    except Exception as e:
        print("Error in /recommend:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_model():
    try:
        print("Incoming request to /ask")
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # System prompt to guide the model's behavior
        system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful and ethical assistant. "
                "Kepp your answers concise and relevant. "
                "Your goal is to assist the user in a friendly manner. "
                "Do not answer questions that are harmful, illegal, offensive, or nonsensical. "
                "If the question is unclear or unsafe, politely refuse to answer."
            )
        }

        # Send to the model
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                system_prompt,
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=512,
        )

        response = completion.choices[0].message.content
        return jsonify({"question": question, "answer": response})

    except Exception as e:
        print("Error in /ask:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
