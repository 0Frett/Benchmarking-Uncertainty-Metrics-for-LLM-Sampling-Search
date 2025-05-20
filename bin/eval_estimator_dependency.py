import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from typing import List, Union, Optional
from data_structs import GT_Tree, Estimator_Tree
import general_utils
import metric_utils
from scipy.stats import spearmanr
import logging
import ipdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

ESTIMATOR_KEYS = [
    "NPE", "LNPE", "SE", "VC_Neg", "PTrue_Comp", "Lexical", "IPT_EU", "SPUQ_Comp", 
    "ANSU", "CU", "Ensemble_AU", "Ensemble_EU"
]


def bootstrap_pairs(
    estimator_tree_cache: Estimator_Tree, 
) -> pd.DataFrame:
    
    tree_keys = list(estimator_tree_cache.keys())
    # bootstrap_tree_keys = np.random.choice(tree_keys, len(tree_keys), replace=True)
    metric_pairs = []
    for key in tree_keys:
        estimator_tree = estimator_tree_cache[key]
        pairs = {}
        for key in ESTIMATOR_KEYS:
            pairs[key] = estimator_tree.metadata[key]
        pairs['ACC'] = estimator_tree.metadata['ACC']
        metric_pairs.append(pairs)
    # df = pd.DataFrame(metric_pairs)
    full_df = pd.DataFrame(metric_pairs)
    full_df = full_df.dropna(axis=1) 
    logging.info(full_df.isna().sum().sum())
    # Now bootstrap from this cleaned set
    bootstrap_indices = np.random.choice(len(full_df), len(full_df), replace=True)

    return full_df.iloc[bootstrap_indices].reset_index(drop=True)


def compute_ci_matrices(
    matrices: List[np.ndarray],
    columns: List[str],
    metric_name: str,
    save_dir: str
) -> None:
    """
    Compute mean, 2.5th and 97.5th percentile matrices and save to CSV.
    """
    array = np.stack(matrices, axis=0)  # shape: (n_bootstrap, n_vars, n_vars)
    mean = np.mean(array, axis=0)
    lower = np.percentile(array, 2.5, axis=0)
    upper = np.percentile(array, 97.5, axis=0)

    mean_df = pd.DataFrame(mean, index=columns, columns=columns)
    lower_df = pd.DataFrame(lower, index=columns, columns=columns)
    upper_df = pd.DataFrame(upper, index=columns, columns=columns)

    mean_df.to_csv(os.path.join(save_dir, f"{metric_name}_bootstrap.csv"))
    lower_df.to_csv(os.path.join(save_dir, f"{metric_name}_pctl2.5.csv"))
    upper_df.to_csv(os.path.join(save_dir, f"{metric_name}_pctl97.5.csv"))
    logging.info(f"[INFO] Saved {metric_name} confidence interval matrices to {save_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--etree_dir', type=str, required=True, help="Directory of Labeled Estimator Tree.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save eval result.")
    parser.add_argument('--n_bootstrap', type=int, default=1000, help="Number of bootstrap samples.")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(f"[INFO] Loading trees from {args.etree_dir} ...")
    estimator_tree_cache = general_utils.create_tree_cache(args.etree_dir)
    

    corr_matrices, mi_matrices = [], []
    logging.info("[INFO] Start Bootstrapping ...")
    for _ in tqdm(range(args.n_bootstrap), desc="Bootstrapping Run"):
        sample_df = bootstrap_pairs(estimator_tree_cache)
        # ipdb.set_trace()
        # sample_df = sample_df.dropna()
        # ipdb.set_trace()
        mi_matrix = metric_utils.mutual_info_matrix_knn(sample_df.to_numpy())
        # corr_matrix = sample_df.corr(method='spearman').to_numpy()
        rho, _ = spearmanr(sample_df, axis=0, nan_policy='omit')
        corr_matrix = rho
        mi_matrices.append(mi_matrix)
        corr_matrices.append(corr_matrix)

    columns = sample_df.columns
    logging.info(f"[INFO] Computing Spearman correlation confidence interval ...")
    compute_ci_matrices(corr_matrices, columns, metric_name="spearman_corr", save_dir=args.save_dir)

    logging.info(f"[INFO] Computing mutual information confidence interval ...")
    compute_ci_matrices(mi_matrices, columns, metric_name="mutual_info", save_dir=args.save_dir)
    logging.info(f"[INFO] All confidence interval matrices saved to {args.save_dir}.")


if __name__ == '__main__':
    main()
