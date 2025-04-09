import numpy as np
import json

with open('valid.json', 'r') as f:
    data = json.load(f)

# extract questions and create indices
questions = []
question_ids = []
contexts = []

for item in data['data']:
    for para in item['paragraphs']:
        for qa in para['qas']:
            questions.append(qa['question'])
            question_ids.append(qa['id'])
            contexts.append(para['context'])

print(questions)
# print(contexts)