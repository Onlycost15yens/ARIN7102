import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math

# Load model
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Load dataset
with open('valid.json', 'r') as f:
    data = json.load(f)

questions, question_ids, contexts, answers = [], [], [], []

for item in data['data']:
    for para in item['paragraphs']:
        for qa in para['qas']:
            questions.append(qa['question'])
            question_ids.append(qa['id'])
            contexts.append(para['context'])
            if not qa['is_impossible'] and len(qa['answers']) > 0:
                answers.append(qa['answers'][0]['text'])
            else:
                answers.append("")

# Build FAISS index
question_embeddings = model.encode(questions)
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings).astype('float32'))

# BM25 & TF-IDF corpus preparation
tokenized_questions = [q.lower().split() for q in questions]
bm25 = BM25Okapi(tokenized_questions)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

# Metrics
def recall_at_k(retrieved_ids, true_id, k):
    return int(true_id in retrieved_ids[:k])

def dcg(relevance_scores):
    return sum([
        rel / math.log2(idx + 2)  # log2(idx+2) because index starts at 0
        for idx, rel in enumerate(relevance_scores)
    ])

def ndcg_at_k(retrieved_ids, true_id, k):
    relevance = [1 if rid == true_id else 0 for rid in retrieved_ids[:k]]
    dcg_val = dcg(relevance)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg_val = dcg(ideal_relevance)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def average_precision_at_k(retrieved_ids, true_id, k):
    hits = 0
    sum_precisions = 0.0
    for i in range(min(k, len(retrieved_ids))):
        if retrieved_ids[i] == true_id:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / hits if hits > 0 else 0.0

def evaluate(method, k=5):
    recalls, ndcgs, maps = [], [], []
    for i, query in enumerate(tqdm(questions[:500])):  # limit for speed
        true_id = question_ids[i]
        
        if method == 'faiss':
            query_vec = model.encode([query])
            _, indices = index.search(np.array(query_vec).astype('float32'), k)
            retrieved_ids = [question_ids[idx] for idx in indices[0]]
        
        elif method == 'bm25':
            scores = bm25.get_scores(query.lower().split())
            top_k_idx = np.argsort(scores)[::-1][:k]
            retrieved_ids = [question_ids[idx] for idx in top_k_idx]
        
        elif method == 'tfidf':
            query_vec = tfidf_vectorizer.transform([query])
            cosine_similarities = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()
            top_k_idx = np.argsort(cosine_similarities)[::-1][:k]
            retrieved_ids = [question_ids[idx] for idx in top_k_idx]
        
        recalls.append(recall_at_k(retrieved_ids, true_id, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, true_id, k))
        maps.append(average_precision_at_k(retrieved_ids, true_id, k))
    
    return {
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs),
        'map': np.mean(maps)
    }

# Run evaluation for each method and metric
methods = ['faiss', 'bm25', 'tfidf']
ks = [1, 3, 5, 10]
metrics = ['recall', 'ndcg', 'map']

results = {metric: {method: [] for method in methods} for metric in metrics}

for method in methods:
    for k in ks:
        scores = evaluate(method, k)
        for metric in metrics:
            results[metric][method].append(scores[metric])

# Plot metrics
colors = {'faiss': 'blue', 'bm25': 'green', 'tfidf': 'red'}
os.makedirs("figures", exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(ks, results[metric][method],
                 label=method.upper(), marker='o', color=colors[method])
    plt.title(f"{metric.upper()}@K for Different Retrieval Methods")
    plt.xlabel("K")
    plt.ylabel(f"{metric.upper()}@K")
    plt.xticks(ks)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300)
    plt.show()