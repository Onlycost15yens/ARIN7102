import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def analyze_medquad(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_qa_pairs = []
    
    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']
                qa_id = qa['id']
                # is_impossible = qa.get('is_impossible', False)
                
                all_qa_pairs.append({
                    'id': qa_id,
                    'question': question,
                    'answer': answer,
                    'question_length': len(question.split()),
                    'answer_length': len(answer.split()),
                    # 'is_impossible': is_impossible
                })
    
    total_pairs = len(all_qa_pairs)
    
    # question length
    question_lengths = [qa['question_length'] for qa in all_qa_pairs]
    avg_question_length = np.mean(question_lengths)
    min_question_length = np.min(question_lengths)
    max_question_length = np.max(question_lengths)
    
    # answer length
    answer_lengths = [qa['answer_length'] for qa in all_qa_pairs]
    avg_answer_length = np.mean(answer_lengths)
    min_answer_length = np.min(answer_lengths)
    max_answer_length = np.max(answer_lengths)
    
    # impossible question
    # impossible_count = sum(1 for qa in all_qa_pairs if qa['is_impossible'])
    
    # longest and shortest questios / answers
    shortest_question = min(all_qa_pairs, key=lambda x: x['question_length'])
    longest_question = max(all_qa_pairs, key=lambda x: x['question_length'])
    shortest_answer = min(all_qa_pairs, key=lambda x: x['answer_length'])
    longest_answer = max(all_qa_pairs, key=lambda x: x['answer_length'])
    
    # question types
    question_types = []
    for qa in all_qa_pairs:
        q = qa['question'].lower()
        if q.startswith('what'):
            question_types.append('what')
        elif q.startswith('how'):
            question_types.append('how')
        elif q.startswith('is') or q.startswith('are'):
            question_types.append('yes/no')
        elif q.startswith('can') or q.startswith('could'):
            question_types.append('can/could')
        elif q.startswith('who'):
            question_types.append('who')
        elif q.startswith('when'):
            question_types.append('when')
        elif q.startswith('where'):
            question_types.append('where')
        elif q.startswith('why'):
            question_types.append('why')
        elif q.startswith('which'):
            question_types.append('which')
        else:
            question_types.append('other')
    
    question_type_count = Counter(question_types)
    
    print(f"total pairs: {total_pairs}")
    print(f"average question length: {avg_question_length:.2f} words")
    print(f"question length range: {min_question_length} - {max_question_length} words")
    print(f"average answer length: {avg_answer_length:.2f} words")
    print(f"answer length range: {min_answer_length} - {max_answer_length} words")
    # print(f"impossible count: {impossible_count} ({impossible_count/total_pairs*100:.2f}%)")
    
    print("\ndistribution of question types:")
    for q_type, count in question_type_count.most_common():
        print(f"  {q_type}: {count} ({count/total_pairs*100:.2f}%)")
    
    print("\nshortest question example:")
    print(f"  '{shortest_question['question']}' ({shortest_question['question_length']} words)")
    
    print("\nlongest question example:")
    print(f"  '{longest_question['question']}' ({longest_question['question_length']} words)")
    
    plt.figure(figsize=(15, 10))
    
    # question length dist
    plt.subplot(2, 2, 1)
    plt.hist(question_lengths, bins=20, alpha=0.7, color='blue')
    plt.axvline(avg_question_length, color='r', linestyle='dashed', linewidth=1)
    plt.title('distribution of question length')
    plt.xlabel('number of words')
    plt.ylabel('frequency')
    
    # answer length dist
    plt.subplot(2, 2, 2)
    plt.hist(answer_lengths, bins=20, alpha=0.7, color='green')
    plt.axvline(avg_answer_length, color='r', linestyle='dashed', linewidth=1)
    plt.title('distribution of answer length')
    plt.xlabel('number of words')
    plt.ylabel('frequency')
    
    # pie chart
    plt.subplot(2, 2, 3)
    question_types_dict = dict(question_type_count.most_common(6))
    if 'other' in question_types_dict:
        question_types_dict['other'] = sum(count for q_type, count in question_type_count.items() 
                                         if q_type not in list(question_types_dict.keys())[:5])
    plt.pie(question_types_dict.values(), labels=question_types_dict.keys(), autopct='%1.1f%%')
    plt.title('distribution of question type')
    
    plt.tight_layout()
    plt.savefig('medquad_analysis.png')
    
    return {
        'total_pairs': total_pairs,
        'avg_question_length': avg_question_length,
        'min_question_length': min_question_length,
        'max_question_length': max_question_length,
        'avg_answer_length': avg_answer_length,
        'min_answer_length': min_answer_length,
        'max_answer_length': max_answer_length,
        # 'impossible_count': impossible_count,
        'question_type_count': question_type_count
    }

stats = analyze_medquad('train.json')
print(stats)