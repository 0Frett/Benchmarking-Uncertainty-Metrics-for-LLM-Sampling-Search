import numpy as np
import pickle
import math
import re
from reasoning_tree import Single_QA_Struct
from typing import Optional, Union, Dict, List

def update_tree_npe(node:Single_QA_Struct):
    def npe_per_node(node):
        actions_values = []
        for q_node in node.qa_matches:
            for a_node in q_node.children:
                logprobsum = 0
                for token in a_node.tokens_prob:
                    logprobsum += token['top1Logprob']
                actions_values.append(logprobsum)
        assert len(actions_values) > 0
        node.npe = -np.mean(actions_values)
    npe_per_node(node)
    
    for child in node.children:
        update_tree_npe(child)

def update_tree_lnpe(node:Single_QA_Struct):
    def lnpe_per_node(node):
        actions_values = []
        for q_node in node.qa_matches:
            for a_node in q_node.children:
                logprobsum = 0
                tokennum = 0
                for token in a_node.tokens_prob:
                    logprobsum += token['top1Logprob']
                    tokennum += 1
                actions_values.append(logprobsum/tokennum)
        node.lnpe = -np.mean(actions_values)

    lnpe_per_node(node)
    for child in node.children:
        update_tree_lnpe(child)   

def update_tree_top2disparity(node:Single_QA_Struct):
    def top2disparity_per_node(node):
        actions_values = []
        for q_node in node.qa_matches:
            for a_node in q_node.children:
                disparity_sum = 0
                tokennum = 0
                for token in a_node.tokens_prob:
                    disparity_sum += (token['top1Logprob']-token['top2Logprob'])
                    tokennum += 1
                actions_values.append(disparity_sum/tokennum)
        node.top2disparity = -np.mean(actions_values)
    
    top2disparity_per_node(node)
    for child in node.children:
        update_tree_top2disparity(child) 

def update_tree_se(node):
    pass

def update_tree_spuq(node):
    pass

def update_tree_verbal_confidence(node):
    pass

def update_tree_intra_sample_score(node):
    pass

def update_tree_fidelity_confidence(node):
    pass

def calculate_entropy(values:list):
    if len(values) == 0:
        return 0.0

    total_count = len(values)
    value_counts = {}
    for value in values:
        # if value == True:
        #     print("yes")
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    entropy = 0.0
    for count in value_counts.values():
        probability = count / total_count
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy

def update_tree_ensemble_approx_answer_uncertainty(root:Single_QA_Struct):
    def collect_node_subtree_answers(node:Single_QA_Struct):
        if not node:
            return []
        
        subtree_answers = []
        for q_node in node.qa_matches:
            for a_node in q_node.children:
                if a_node.is_terminate:
                    subtree_answers.append(a_node.answer)

        for child in node.children:
             # Add scores from the child subtree
            subtree_answers.extend(collect_node_subtree_answers(child)) 
        node.subtree_answers = subtree_answers

        return subtree_answers
    
    def count_single_hypothesis_label_probs(values:list, answer_set:set):
        total = len(values)
        label_prob_dict = {}
        for ans in answer_set:
            label_prob_dict[ans] = values.count(ans)/total
        return label_prob_dict
        

    def update_node_ensemble_approx_uncertainty(node:Single_QA_Struct, answer_set:set):
        if not node:
            return

        entropy_for_each_hypothesis = []
        label_probs_of_each_hypothesis = {ans:[] for ans in answer_set}
        if node.children == []:
            # end nodes still have hidden children
            for q_node in node.qa_matches:
                h_nodes = [a_node.answer for a_node in q_node.children]
                prob_dict = count_single_hypothesis_label_probs(h_nodes, answer_set)
                for key, value in prob_dict.items():
                    label_probs_of_each_hypothesis[key].append(value)
                entropy_for_each_hypothesis.append(calculate_entropy(h_nodes))
        else:
            for child in node.children:
                h_nodes = child.subtree_answers
                prob_dict = count_single_hypothesis_label_probs(h_nodes, answer_set)
                for key, value in prob_dict.items():
                    label_probs_of_each_hypothesis[key].append(value)
                entropy_for_each_hypothesis.append(calculate_entropy(h_nodes))
        
        # update node values
        node.ensemble_approx_answer_aleatoric_uncertainty = np.mean(entropy_for_each_hypothesis)
        hypotheses_average_label_probs = [
            np.mean(hypotheses_label_prob) for hypotheses_label_prob in label_probs_of_each_hypothesis.values()
        ]
        entropy = 0.0
        for probability in hypotheses_average_label_probs:
            if probability >0:
                entropy -= probability * math.log2(probability)
        node.ensemble_approx_answer_total_uncertainty = entropy
        #print(node.answer_total_uncertainty, node.ensemble_approx_answer_total_uncertainty)
        node.ensemble_approx_answer_epistemic_uncertainty = (
            node.ensemble_approx_answer_total_uncertainty - node.ensemble_approx_answer_aleatoric_uncertainty
        )

        for child in node.children:
            update_node_ensemble_approx_uncertainty(child, answer_set)

    all_answers = collect_node_subtree_answers(root)
    answer_set = set(all_answers)
    update_node_ensemble_approx_uncertainty(root, answer_set)

def update_tree_ensemble_approx_correctness_uncertainty(root:Single_QA_Struct):
    def collect_node_subtree_correctness(node:Single_QA_Struct):
        if not node:
            return []
        
        subtree_correctness = []
        for q_node in node.qa_matches:
            for a_node in q_node.children:
                if a_node.is_terminate:
                    subtree_correctness.append(a_node.is_correct)

        for child in node.children:
             # Add scores from the child subtree
            subtree_correctness.extend(collect_node_subtree_correctness(child)) 
        node.subtree_correctness = subtree_correctness

        return subtree_correctness

    def update_node_ensemble_approx_uncertainty(node:Single_QA_Struct):
        if not node:
            return
        entropy_for_each_hypothesis = []
        correct_prob_for_each_hypothesis = []
        wrong_prob_for_each_hypothesis = []
        if node.children == []:
            # end nodes still have hidden children
            for q_node in node.qa_matches:
                h_nodes = [a_node.is_correct for a_node in q_node.children]
                correct_prob_for_each_hypothesis.append(h_nodes.count(True)/len(h_nodes))
                wrong_prob_for_each_hypothesis.append(h_nodes.count(False)/len(h_nodes))
                entropy_for_each_hypothesis.append(calculate_entropy(h_nodes))
        else:
            for child in node.children:
                h_nodes = child.subtree_correctness
                correct_prob_for_each_hypothesis.append(h_nodes.count(True)/len(h_nodes))
                wrong_prob_for_each_hypothesis.append(h_nodes.count(False)/len(h_nodes))
                entropy_for_each_hypothesis.append(calculate_entropy(h_nodes))

        # update node values
        node.ensemble_approx_correctness_aleatoric_uncertainty = np.mean(entropy_for_each_hypothesis)
        hypotheses_average_label_probs = [
            np.mean(correct_prob_for_each_hypothesis), 
            np.mean(wrong_prob_for_each_hypothesis)
        ]
        entropy = 0.0
        for probability in hypotheses_average_label_probs:
            if probability > 0:
                entropy -= probability * math.log2(probability)
        node.ensemble_approx_correctness_total_uncertainty = entropy

        # print(node.correctness_total_uncertainty, node.ensemble_approx_correctness_total_uncertainty)
        #assert node.correctness_total_uncertainty == node.ensemble_approx_correctness_total_uncertainty
        node.ensemble_approx_correctness_epistemic_uncertainty = (
            node.ensemble_approx_correctness_total_uncertainty - node.ensemble_approx_correctness_aleatoric_uncertainty
        )

        for child in node.children:
            update_node_ensemble_approx_uncertainty(child)

    tree_correctness = collect_node_subtree_correctness(root)
    print('correct nodes :', tree_correctness.count(True))
    print('wrong nodes :', tree_correctness.count(False))
    update_node_ensemble_approx_uncertainty(root)

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_tree(root, filename):
    with open(filename, 'wb') as f:
        pickle.dump(root, f)


if __name__ == '__main__':
    for idx in [8, 39, 74, 107, 119, 141, 144]:#,,, 39,218, 154, 161, 177, 211, 214, 
        print(f"============{idx}===============")
        filename = f'./output_trees/Q{idx}/tree_1/tree.pkl'
        root = load_tree(filename=filename)
        update_tree_npe(node=root)
        update_tree_lnpe(node=root)
        update_tree_top2disparity(node=root)
        # update_tree_se(node=root)
        # update_tree_spuq(node=root)
        # update_tree_verbal_confidence(node=root)
        # update_tree_intra_sample_score(node=root)
        # update_tree_fidelity_confidence(node=root)
        update_tree_ensemble_approx_correctness_uncertainty(root=root)
        update_tree_ensemble_approx_answer_uncertainty(root=root)

        save_tree(root=root, filename=filename)

