import numpy as np
import pickle
import math
import pandas as pd
from collections import deque
import os
import re

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_metric_multiple_trees_df(qindices, tree_nums, savename):
    metrics = []
    for idx in qindices: #,
        for tree_num in tree_nums:
            try:
                filename = f'./output_nodes/Q{idx}/node_{tree_num}/node.pkl'
                node = load_tree(filename=filename)
                metrics.append(
                    [
                        node.npe, 
                        node.lnpe, 
                        node.top2disparity,
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
    df.to_csv(f'./{savename}.csv', index=False)


if __name__ == '__main__':

    qindices = [int(re.search(r'\d+', s).group()) for s in os.listdir('./output_nodes')]
    print(len(qindices))
    #qindices = [2, 8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]
    tree_nums = [i for i in range(6)]
    get_all_metric_multiple_trees_df(qindices, tree_nums, 'metrics_stats')
    


