# 📚 PaperHub

**PaperHub** is a scalable hybrid research paper recommender system that intelligently matches scholarly queries with relevant papers using a combination of semantic similarity (via Sentence-BERT) and citation-based scoring. It supports both category-based and cluster-based filtering using a trained KMeans model.

---

## ✨ Features

- 🔍 Query understanding using Sentence-BERT embeddings
- 🧠 Hybrid scoring using citation count + semantic similarity
- 📦 Optional KMeans-based cluster filtering
- 🖼️ Visually rich, responsive frontend built with Next.js + TailwindCSS
- 💾 Bookmark functionality using local storage
- 📄 PDF and source links for each recommended paper

---

## 🚀 Getting Started

To run the full stack locally, follow the instructions below.

---

## 📁 1. Fork and Clone the Repository

```sh
git clone https://github.com/your-username/PaperHub.git
📦 2. Install Dependencies
🔧 Backend (Python)
sh
Copy
Edit
pip install -r requirements.txt
🌐 Frontend (React + Next.js)
sh
Copy
Edit
cd paperhub-ui
npm install
🧠 3. Run the App
▶️ Start the Backend Server
From the root of the project:

sh
Copy
Edit
python app.py
💻 Start the Frontend
sh
Copy
Edit
cd paperhub-ui
npm run dev
Now visit 👉 http://localhost:3000 in your browser to explore the app.

