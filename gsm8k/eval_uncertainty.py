import numpy as np
import pickle
import math
import pandas as pd
from collections import deque
import os
import re
import random
from tqdm import tqdm

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_metric_multiple_trees_df(qindices, tree_nums, savename, modelname):
    metrics = []
    for idx in tqdm(qindices, desc="Processing qindices"):
        for tree_num in tree_nums:
            # try:
            filename = f'./output_nodes/{modelname}/Q{idx}/node_{tree_num}/node.pkl'
            node = load_tree(filename=filename)
            metrics.append(
                [
                    node.npe, 
                    node.lnpe, 
                    node.top2disparity,
                    node.semantic_entropy,
                    node.verb_conf,
                    node.lexical_similarity,
                    node.ensemble_approx_answer_total_uncertainty,
                    node.ensemble_approx_correctness_total_uncertainty,
                    node.ensemble_approx_answer_aleatoric_uncertainty,
                    node.ensemble_approx_answer_epistemic_uncertainty,
                    node.correct_ratio
                    # node.inter_sample_score,
                    # node.ensemble_approx_correctness_aleatoric_uncertainty,
                    # node.ensemble_approx_correctness_epistemic_uncertainty,
                    # node.verb_predict_performance,
                ]
            )
                #print(metrics)
            # except:
            #     print(f"Skipping index {idx} tree {tree_num} due to error")
            #     continue

    df = pd.DataFrame(
        metrics, 
        columns=[
            'NPE', 
            'LNPE', 
            'Top-DISP',
            'SE',
            'Intra',
            'Lexical',
            'AnsU',
            'CU',
            'AU',
            'EU',
            'ACC'
            # 'inter_sample',
            # 'correctness_aleatoric',
            # 'correctness_epistemic',
            # 'verb_confidence',
            # 'verb_predict_acc',
        ]
    )
    df.to_csv(f'./node_stats/{savename}.csv', index=False)  #metrics_df/
    return df


def bootstrap_corr(df, n_bootstrap=1000):
    """
    Perform bootstrap sampling on the DataFrame to compute a distribution of correlation matrices.
    Returns:
      mean_corr_df: DataFrame of the mean correlation coefficients.
      lower_corr_df: DataFrame of the 2.5th percentile for each coefficient.
      upper_corr_df: DataFrame of the 97.5th percentile for each coefficient.
    """
    corr_matrices = []
    # Using tqdm to monitor progress over bootstrap iterations
    for i in tqdm(range(n_bootstrap), desc="Bootstrapping correlations"):
        # Sample rows with replacement from the DataFrame
        sample_df = df.sample(frac=1.0, replace=True)
        corr = sample_df.corr()
        corr_matrices.append(corr.values)
    
    # Convert list of matrices into a numpy array: shape (n_bootstrap, n_features, n_features)
    corr_array = np.stack(corr_matrices, axis=0)
    
    # Compute mean correlation matrix and percentiles
    mean_corr = np.mean(corr_array, axis=0)
    lower_corr = np.percentile(corr_array, 2.5, axis=0)
    upper_corr = np.percentile(corr_array, 97.5, axis=0)
    
    columns = df.columns
    mean_corr_df = pd.DataFrame(mean_corr, index=columns, columns=columns)
    lower_corr_df = pd.DataFrame(lower_corr, index=columns, columns=columns)
    upper_corr_df = pd.DataFrame(upper_corr, index=columns, columns=columns)
    
    return mean_corr_df, lower_corr_df, upper_corr_df

def compute_pvalues(df, n_permutations=1000):
    """
    Compute p-values for each pairwise correlation coefficient in the DataFrame using permutation tests.
    Returns a DataFrame of p-values.
    """
    columns = df.columns
    p_values = np.zeros((len(columns), len(columns)))
    
    # Outer loop with progress bar
    for i in tqdm(range(len(columns)), desc="Permutation tests (outer loop)"):
        for j in range(i, len(columns)):
            if i == j:
                p_values[i, j] = 0.0
            else:
                x = df[columns[i]].values
                y = df[columns[j]].values
                # Compute observed correlation
                r_obs = np.corrcoef(x, y)[0, 1]
                count = 0
                # Inner loop with progress bar (nested, leave=False to avoid clutter)
                for _ in tqdm(range(n_permutations), leave=False, desc=f"Permuting {columns[i]} vs {columns[j]}"):
                    # Permute y to break any association
                    y_perm = np.random.permutation(y)
                    r_perm = np.corrcoef(x, y_perm)[0, 1]
                    if abs(r_perm) >= abs(r_obs):
                        count += 1
                p_val = count / n_permutations
                p_values[i, j] = p_val
                p_values[j, i] = p_val  # Symmetric matrix
                
    p_value_df = pd.DataFrame(p_values, index=columns, columns=columns)
    return p_value_df

if __name__ == '__main__':
    os.makedirs('./node_stats', exist_ok=True)
    for modelname in ['gemma', 'llama', 'gpt-3.5-turbo']:
        # Get qindices from the folder names in the output directory
        qindices = [int(re.search(r'\d+', s).group()) for s in os.listdir(f'./output_nodes/{modelname}')]
        # Collect metrics and create a DataFrame
        df = get_all_metric_multiple_trees_df(qindices, [1], f'{modelname}_metrics_stats', modelname)
        
        # Perform bootstrapping to compute the correlation matrix and its confidence intervals
        mean_corr_df, lower_corr_df, upper_corr_df = bootstrap_corr(df, n_bootstrap=300)
        
        # Save the bootstrapped correlation matrices
        mean_corr_df.to_csv(f'./node_stats/{modelname}_metrics_mean_correlation.csv')
        lower_corr_df.to_csv(f'./node_stats/{modelname}_metrics_lower_corr_2.5_percentile.csv')
        upper_corr_df.to_csv(f'./node_stats/{modelname}_metrics_upper_corr_97.5_percentile.csv')
        
        # Compute p-values for the correlation coefficients using permutation tests
        p_value_df = compute_pvalues(df, n_permutations=300)
        p_value_df.to_csv(f'./node_stats/{modelname}_metrics_correlation_pvalues.csv', index=True)
        
        print("Bootstrapping and permutation tests complete. Correlation matrices and p-values saved.")
