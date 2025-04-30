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
```
---
📦 2. Install Dependencies
🔧 Backend (Python)
```sh
pip install -r requirements.txt
```
---
🌐 3. Frontend (React + Next.js)
```sh
cd paperhub-ui
npm install
```
---
🧠 4. Run the App
▶️ Start the Backend Server
From the root of the project:
```sh
python app.py
```
---
💻 5. Start the Frontend
```sh
cd paperhub-ui
npm run dev
```
Now visit 👉 http://localhost:3000 in your browser to explore the app.
![image](https://github.com/user-attachments/assets/e4e30e67-c26f-4b1d-a111-cab3913336ad)
## Recommendations:
![image](https://github.com/user-attachments/assets/c628629d-9cd5-4a0c-956d-d16ec4738fef)

![image](https://github.com/user-attachments/assets/d5d13a4a-a0d2-4ddc-89c4-6dd5cfde4a43)



