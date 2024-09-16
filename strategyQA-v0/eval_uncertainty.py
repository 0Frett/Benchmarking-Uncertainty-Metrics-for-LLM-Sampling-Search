import numpy as np
import pickle
import math
import pandas as pd
from collections import deque
import os
import re
from hard_qs import hard1, hard2, hard3, hard4, hard5

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_metric_multiple_trees_df(qindices, tree_nums, savename, model_name):
    metrics = []
    for idx in qindices: #,
        for tree_num in tree_nums:
            try:
                filename = f'./output_nodes/{model_name}/Q{idx}/node_{tree_num}/node.pkl'
                node = load_tree(filename=filename)
                metrics.append(
                    [
                        node.npe, 
                        node.lnpe, 
                        node.top2disparity,
                        node.inter_sample_score,
                        node.ensemble_approx_answer_total_uncertainty,
                        node.ensemble_approx_answer_aleatoric_uncertainty,
                        node.ensemble_approx_answer_epistemic_uncertainty,
                        node.ensemble_approx_correctness_total_uncertainty,
                        node.ensemble_approx_correctness_aleatoric_uncertainty,
                        node.ensemble_approx_correctness_epistemic_uncertainty,
                        node.verb_conf,
                        node.verb_predict_performance,
                        node.correct_ratio
                    ]
                )
                #print(metrics)
            except:
                continue
    df = pd.DataFrame(
        metrics, 
        columns=[
            'npe', 
            'lnpe', 
            'top2disparity',
            'inter_sample',
            'answer_total',
            'answer_aleatoric',
            'answer_epistemic',
            'correctness_total',
            'correctness_aleatoric',
            'correctness_epistemic',
            'verb_confidence',
            'verb_predict_acc',
            'correct_ratio'
        ]
    )
    df.to_csv(f'./metrics_df/{savename}.csv', index=False)


if __name__ == '__main__':
    model = "llama"
    qindices = [int(re.search(r'\d+', s).group()) for s in os.listdir(f'./output_nodes/{model}')]
    run_indices = [idx for idx in hard1 if idx in qindices]
    # for idxx in qindices:
    tree_nums = [1]
    get_all_metric_multiple_trees_df(run_indices, tree_nums, f'{model}_h1_metrics_stats', model)
    


