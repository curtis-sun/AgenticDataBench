import json
import os

STEP_PATTERN = r'\*\*(.+?)\*\*(?:\s*\([^)]*\))?\s*:\s*(.*)'

def compute_stackoverflow_idx(obj):
    return f'{obj['id']}-{obj['answer_id']}'

def read_idxes(filename, compute_idx = lambda obj: obj['id']):
    idxes = []
    if os.path.exists(filename):
        with open(filename, 'r') as fin:
            for line in fin:
                obj = json.loads(line)
                idxes.append(compute_idx(obj))
    return idxes

def prepare_qas(input_file, output_file):
    qas = []
    with open(input_file, 'r') as fin:
        for line in fin:
            obj = json.loads(line)
            obj['index'] = compute_stackoverflow_idx(obj)
            qas.append(obj)
    finished_idxes = read_idxes(output_file, compute_stackoverflow_idx)
    qas = [obj for obj in qas if obj['index'] not in finished_idxes]
    return qas

def prepare_clusters(input_file, output_file):
    docs = []
    with open(input_file, 'r') as fin:
        for line in fin:
            cluster = json.loads(line)
            docs.append({'id': str(len(docs)), 'cluster': cluster})
    finished_idxes = read_idxes(output_file)
    docs = [obj for obj in docs if obj['id'] not in finished_idxes]
    return docs
