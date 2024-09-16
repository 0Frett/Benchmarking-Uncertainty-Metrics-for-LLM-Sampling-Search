import numpy as np
import pickle
import math
import re
from llms_parallel import OpenAIModel

def calculate_npe(node):
    if not node.children:
        node.npe = 0.0  # No children, metric is zero
        return
    
    children_values = []
    for child in node.children:
        logprobsum = 0
        tokennum = 0
        for token in child.action_tokens:
            logprobsum += token['top1Logprob']
            tokennum += 1
        children_values.append(logprobsum)
    node.npe = -np.mean(children_values)
    
    for child in node.children:
        calculate_npe(child)

def calculate_lnpe(node):
    if not node.children:
        node.lnpe = 0.0  # No children, metric is zero
        return
    
    children_values = []
    for child in node.children:
        logprobsum = 0
        tokennum = 0
        for token in child.action_tokens:
            logprobsum += token['top1Logprob']
            tokennum += 1
        children_values.append(logprobsum/tokennum)
    node.lnpe = -np.mean(children_values)
    
    for child in node.children:
        calculate_lnpe(child)

def calculate_top2disparity(node):
    if not node.children:
        node.top2disparity = 0.0  # No children, metric is zero
        return
    
    children_values = []
    for child in node.children:
        disparity_sum = 0
        tokennum = 0
        for token in child.action_tokens:
            disparity_sum += (token['top1Logprob'] - token['top2Logprob'])
            tokennum += 1

        children_values.append(disparity_sum/tokennum)
    node.top2disparity = -np.mean(children_values)
    
    for child in node.children:
        calculate_top2disparity(child)


def calculate_answer_total_uncertainty(root):
    def calculate_entropy(values:list):
        if len(values) == 0:
            return 0.0

        total_count = len(values)
        value_counts = {}
        for value in values:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
        
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        
        return entropy

    def calculate_entropies(node):
        if not node.children: # is leaf
            numbers = re.findall(r'\d+', node.answer)
            # if len(numbers) > 1:
            #     print(numbers)
            if len(numbers) == 0:
                node.answer = "Fail"
            else:
                node.answer = numbers[0]
            node.leaf_values = [node.answer]
            node.answer_entropy = calculate_entropy(node.leaf_values)
            return [node.answer]

        all_leaf_values = []
        for child in node.children:
            child_leaf_values = calculate_entropies(child)
            all_leaf_values.extend(child_leaf_values)
            
        node.leaf_values = all_leaf_values.copy()
        node.answer_entropy = calculate_entropy(node.leaf_values)

        return all_leaf_values
    
    calculate_entropies(root)

def calculate_correctness_total_uncertainty(root):
    def calculate_entropy(values:list):
        if len(values) == 0:
            return 0.0

        total_count = len(values)
        value_counts = {}
        for value in values:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
        
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        
        return entropy

    def calculate_entropies(node):
        if not node.children: # is leaf
            correctness = 1 if node.correct else 0
            node.leaf_values = [correctness]
            node.correctness_entropy = calculate_entropy(node.leaf_values)
            node.correct_ratio = node.leaf_values.count(1)/len(node.leaf_values)
            return [correctness]

        all_leaf_values = []
        for child in node.children:
            child_leaf_values = calculate_entropies(child)
            all_leaf_values.extend(child_leaf_values)
            
        node.leaf_values = all_leaf_values.copy()
        node.correct_ratio = node.leaf_values.count(1)/len(node.leaf_values)
        node.correctness_entropy = calculate_entropy(node.leaf_values)

        return all_leaf_values
    
    calculate_entropies(root)


def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_tree(root, filename):
    with open(filename, 'wb') as f:
        pickle.dump(root, f)


if __name__ == '__main__':
    for idx in [8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]:#,
        print(f"============{idx}===============")
        filename = f'./output_trees/Q{idx}/tree_1/tree.pkl'
        root = load_tree(filename=filename)
        calculate_npe(node=root)
        calculate_lnpe(node=root)
        calculate_top2disparity(node=root)
        calculate_answer_total_uncertainty(root=root)
        calculate_correctness_total_uncertainty(root=root)
        save_tree(root=root, filename=filename)

