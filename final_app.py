import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import faiss
import pickle
arxiv_categories = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation and Language",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.SE": "Software Engineering",
    "cs.CY": "Computers and Society",
    "cs.CR": "Cryptography and Security",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.MA": "Multiagent Systems",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.AR": "Hardware Architecture",
    "cs.OS": "Operating Systems",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.GT": "Game Theory",
    "cs.GR": "Graphics",
    "cs.MM": "Multimedia",
    "cs.SD": "Sound",
    "cs.SC": "Symbolic Computation",
    "cs.CC": "Computational Complexity",
    "cs.LO": "Logic in Computer Science",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",
    "quant-ph": "Quantum Physics"
}






def build_citation_graph(data):
    citation_graph = defaultdict(int)
    for _, row in data.iterrows():
        paper_id = row['id']
        citations = row.get('citations', 0)
        citation_graph[paper_id] = citations
    return citation_graph

def citation_similarity(citation_graph, paper_id, other_paper_id):
    return citation_graph[paper_id] / (citation_graph[other_paper_id] + 1)

def recommend_papers_by_keywords(
    data,
    keywords,
    top_n=5,
    citation_graph=None,
    model_path='models\\best_kmeans_model.pkl',
    use_kmeans_filtering=True  # <-- toggle this flag to switch logic
):
    if citation_graph is None:
        citation_graph = build_citation_graph(data)

    # Load BERT model
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Match Query to Closest arXiv Category
    category_texts = list(arxiv_categories.values())
    category_codes = list(arxiv_categories.keys())
    category_embeddings = bert_model.encode(category_texts, convert_to_tensor=False)

    query_vector = bert_model.encode([keywords], convert_to_tensor=False)
    query_vector = np.array(query_vector).reshape(1, -1)

    # Compute similarity with category embeddings
    similarities = cosine_similarity(query_vector, np.array(category_embeddings))
    best_category_index = np.argmax(similarities)
    predicted_category_code = category_codes[best_category_index]
    predicted_category_name = arxiv_categories[predicted_category_code]

    print(f"Predicted Category      : {predicted_category_name}")
    print(f"Predicted Category Code : {predicted_category_code}")
    print(f"Similarity Score        : {similarities[0][best_category_index]:.4f}")

    if use_kmeans_filtering:
        # Step 2a: Predict cluster using saved KMeans model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KMeans model not found at: {model_path}")
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)

        category_vector = bert_model.encode([predicted_category_code], convert_to_tensor=False)
        predicted_cluster = kmeans.predict(np.array(category_vector))[0]
        print(f"📦 Predicted Cluster ID    : {predicted_cluster}")

        # Step 3a: Predict cluster for each paper based on its category
        paper_category_embeddings = bert_model.encode(data['category'].tolist(), convert_to_tensor=False)
        paper_clusters = kmeans.predict(np.array(paper_category_embeddings))
        indices_in_cluster = np.where(paper_clusters == predicted_cluster)[0]
        filtered_data = data.iloc[indices_in_cluster].reset_index(drop=True)
    else:
        # Step 2b: Filter directly by predicted category
        filtered_data = data[data['category'] == predicted_category_code].reset_index(drop=True)

    if filtered_data.empty:
        print("No papers found in filtered data — falling back to full dataset.")
        filtered_data = data.copy()

    # Step 4: Encode titles for FAISS similarity
    title_embeddings = bert_model.encode(filtered_data['title'], convert_to_tensor=False)
    combined_features = np.array(title_embeddings)

    # Step 5: FAISS index
    dimension = combined_features.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(combined_features.astype(np.float32))

    _, indices = faiss_index.search(query_vector.astype(np.float32), top_n)

    # Step 6: Score and return
    hybrid_scores = {}
    for idx in indices[0]:
        paper_id = filtered_data.iloc[idx]['id']
        content_score = cosine_similarity(query_vector, combined_features[idx].reshape(1, -1))[0][0]
        citation_score = np.mean([
            citation_similarity(citation_graph, paper_id, other_id)
            for other_id in filtered_data['id'] if other_id != paper_id
        ])
        hybrid_scores[paper_id] = 0.8 * content_score + 0.2 * citation_score

    recommended_papers = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = []

    for paper_id, score in recommended_papers:
        paper = filtered_data[filtered_data['id'] == paper_id].iloc[0]
        recommendations.append({
            'id': paper_id,
            'title': paper['title'],
            'abstract': paper['abstract'],
            'authors': paper['authors'],
            'citations': paper['citations'],
            'category': paper['category'],
        })

    return recommendations