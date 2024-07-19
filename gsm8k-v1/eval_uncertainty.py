import numpy as np
import pickle
import math
import pandas as pd
from collections import deque

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_metric_single_tree_df(root):
    def collect_all_metrics(node, metrics:list):
        if len(node.children) <= 1:
            return
        metrics.append([node.npe, node.lnpe, node.top2disparity, node.answer_entropy, node.correctness_entropy])
        for child in node.children:
            collect_all_metrics(child, metrics)
        
    metrics = []
    collect_all_metrics(root, metrics)
    df = pd.DataFrame(metrics, columns=['npe', 'lnpe', 'top2disparity', 'answer_entropy', 'correctness_entropy'])

    return df

def get_level_metric_single_tree_dfs(root):
    def collect_level_metrics(root):
        if not root:
            return []

        level_dfs = []
        queue = deque([(root, 0)])

        while queue:
            node, level = queue.popleft()

            if len(level_dfs) <= level:
                level_dfs.append([])

            level_dfs[level].append((node.npe, node.lnpe, node.top2disparity, node.answer_entropy, node.correctness_entropy))

            for child in node.children:
                if len(child.children) <= 1:
                    continue
                queue.append((child, level + 1))

        # Convert each level's data to a DataFrame
        level_dataframes = []
        for level, data in enumerate(level_dfs):
            df = pd.DataFrame(data, columns=['npe', 'lnpe', 'top2disparity', 'answer_entropy', 'correctness_entropy'])
            level_dataframes.append(df)

        return level_dataframes

    dfs = collect_level_metrics(root)

    return dfs

def get_all_metric_multiple_trees_df(qindices, tree_nums, savename):
    dfs = []
    for idx in qindices: #,
        for tree_num in tree_nums:
            filename = f'./output_trees/Q{idx}/tree_{tree_num}/tree.pkl'
            root = load_tree(filename=filename)
            metric_df = get_all_metric_single_tree_df(root=root)
            dfs.append(metric_df)
    combined_df = pd.concat(dfs, axis=0)
    combined_df.to_csv(f'./{savename}.csv', index=False)

def get_level_metric_multiple_trees_dfs(qindices, tree_nums, savename):
    level_dfs_multiple_trees = {}
    for qidx in qindices: #,
        for tree_num in tree_nums:
            filename = f'./output_trees/Q{qidx}/tree_{tree_num}/tree.pkl'
            root = load_tree(filename=filename)
            level_dfs_per_tree = get_level_metric_single_tree_dfs(root=root)
            for idx, level_df in enumerate(level_dfs_per_tree):
                if idx not in level_dfs_multiple_trees.keys():
                    level_dfs_multiple_trees[idx] = []
                level_dfs_multiple_trees[idx].append(level_df)
    for level, dfs in level_dfs_multiple_trees.items():
        merge_df = pd.concat(dfs, axis=0).copy()
        merge_df.to_csv(f'./level{level}_{savename}.csv', index=False)


if __name__ == '__main__':
    qindices = [8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]
    tree_nums = [1, 2]
    get_all_metric_multiple_trees_df(qindices, tree_nums, 'metrics_stats')
    get_level_metric_multiple_trees_dfs(qindices, tree_nums, 'metrics_stats')


