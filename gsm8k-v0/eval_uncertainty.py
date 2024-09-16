import numpy as np
import pickle
import math
import pandas as pd
from collections import deque
import os
import re
from hard_qidxs import *

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_metric_multiple_trees_df(qindices, tree_nums, savename, modelname):
    metrics = []
    for idx in qindices: #,
        for tree_num in tree_nums:
            try:
                filename = f'./output_nodes/{modelname}/Q{idx}/node_{tree_num}/node.pkl'
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

    qindices = [int(re.search(r'\d+', s).group()) for s in os.listdir('./output_nodes/gpt-3.5-turbo')]
    # hard_qindices = hard_qs
    # print(len(hard_qindices))
    # print(len(hard_qs))
    # easy_qindices = [item for item in qindices if item not in hard_qindices]
    # print(len(qindices))
    # qindices = [2, 8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]
    modelname = 'gpt-3.5-turbo'
    # idis = [qs_2, qs_3, qs_4, qs_5, qs_6, qs_7, qs_8, qs_9, qs_11]
    # h = [2,3,4,5,6,7,8,9,11]
    # for idxx in range(len(h)):
    #     tree_nums = [1,2]
    #     get_all_metric_multiple_trees_df(idis[idxx], tree_nums, f'{modelname}_r{h[idxx]}_metrics_stats', modelname)

    get_all_metric_multiple_trees_df(qindices, [1], f'gpt-3.5-turbo_metrics_stats', modelname)
    


