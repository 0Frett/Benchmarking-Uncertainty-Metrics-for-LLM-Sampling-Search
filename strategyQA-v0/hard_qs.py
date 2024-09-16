import json
from collections import defaultdict, Counter

data_file_path = "./data/strategyqa_test.json"
with open(data_file_path, 'r') as f:
    dataset = json.load(f)

decomposes = []
decompose_qid_map = defaultdict(list)

for idx, iii in enumerate(dataset):
    decompose_len = len(iii['decomposition'])
    decomposes.append(decompose_len)
    decompose_qid_map[decompose_len].append(idx)

hard1 = decompose_qid_map[1]
hard2 = decompose_qid_map[2]
hard3 = decompose_qid_map[3]
hard4 = decompose_qid_map[4]
hard5 = decompose_qid_map[5]
