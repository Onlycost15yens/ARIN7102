from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# medical expert model
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

with open('valid.json', 'r') as f:
    data = json.load(f)

# extract questions and create indices
questions = []
question_ids = []
contexts = []
answers = [] 

for item in data['data']:
    for para in item['paragraphs']:
        for qa in para['qas']:
            questions.append(qa['question'])
            question_ids.append(qa['id'])
            contexts.append(para['context'])
            
            if not qa['is_impossible'] and len(qa['answers']) > 0:
                answers.append(qa['answers'][0]['text'])
            else:
                answers.append("No answer available")  

# question -> vectors
question_embeddings = model.encode(questions)

# create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(np.array(question_embeddings).astype('float32'))

def retrieve(query, top_k=5):
    query_vector = model.encode([query])
    
    # index the most similar question
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'question': questions[idx],
            'id': question_ids[idx],
            'context': contexts[idx],
            'answer': answers[idx],  
            'score': float(1 - distances[0][i]/100)  # distance -> score
        })
    
    return results

# test
results = retrieve("How many people are affected by arginine:glycine amidinotransferase deficiency ?", top_k=5)
for r in results:
    print(f"Score: {r['score']:.4f}, Question: {r['question']}")
    print(f"Answer: {r['answer']}")
    print("-" * 80)