__author__ = 'Qiao Jin'

import json
from sklearn.metrics import accuracy_score, f1_score, precision_score
import sys

data = sys.argv[1] # pubmedqa or bioasq
pred_path = sys.argv[2]
save_path = pred_path.replace('.json', '.txt')

if data == 'pubmedqa':
    ground_truth = json.load(open('rag/data/pubmedqa/test_ground_truth.json', encoding='utf-8'))
elif data == 'bioasq':
    raw_data = json.load(open('rag/data/bioasq/test_set.json', encoding='utf-8'))
    ground_truth = {}
    for key, values in raw_data.items():
        ground_truth[key] = values['final_decision']

predictions = json.load(open(pred_path, encoding='utf-8'))

assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'

pmids = list(ground_truth)
truth = [ground_truth[pmid].lower() for pmid in pmids]
preds = [predictions[pmid].lower() for pmid in pmids]

acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')
map = precision_score(truth, preds, average='macro')

print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
print('Macro-Precision %f' % map)
with open(save_path, 'a', encoding='utf-8') as file:
    file.write('Accuracy %f' % acc + '\n')
    file.write('Macro-F1 %f' % maf + '\n')
    file.write('Macro-Precision %f' % map)