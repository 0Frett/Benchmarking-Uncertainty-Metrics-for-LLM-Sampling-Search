import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import general_utils
import logging
import math
import ipdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

ESTIMATOR_KEYS = [
    "NPE", "LNPE", "SE", "VC_Neg", "PTrue_Comp", "Lexical", "IPT_EU", "SPUQ_Comp", 
    "ANSU", "CU", "Ensemble_AU", "Ensemble_EU"
]


def scale_eu(x):
    return 1 - 2 ** (-x)

def scale_au(x):
    return x / 2

def scale_ou(x):
    return x / math.log(2)

def scale_su(x):
    return x

def mixture(au, su, eu, ou, method):
    if method == "maxi":
        return max([su, au, eu, ou])
    elif method == "multiply":
        return su*au*eu*ou
    elif method == "mean":
        return (su+au+eu+ou)/4
    else:
        raise ValueError("Unsupported Method")

def bootstrap_pairs(
    estimator_tree_cache: dict,
) -> pd.DataFrame:
    """
    Sample trees with replacement, extract metrics into a DataFrame.
    """

    keys = list(estimator_tree_cache.keys())
    # sampled = np.random.choice(keys, size=len(keys), replace=True)
    records = []
    for k in keys:
        et = estimator_tree_cache[k]
        rec = {}
        # copy estimator metadata
        for m in ESTIMATOR_KEYS:
            rec[m] = et.metadata[m]
        
        # ACC
        rec['ACC'] = et.metadata['ACC']
        records.append(rec)

    full_df = pd.DataFrame(records)
    full_df = full_df.dropna(axis=1) 
    # logging.info(full_df.isna().sum().sum())
    bootstrap_indices = np.random.choice(len(full_df), len(full_df), replace=True)

    return full_df.iloc[bootstrap_indices].reset_index(drop=True)


def compute_bootstrap_metrics(
    df: pd.DataFrame,
    acc_percentile: float
) -> np.ndarray:
    """
    Given a sample DataFrame, compute AUROC, AUPR for each metric column (excluding ACC).
    Returns an array of shape (n_metrics, 2) in order [AUROC, AUPR].
    """
    # threshold for positive class
    if df['ACC'].nunique() == 1:
        logging.warning(f"Uniform ACC = {df['ACC'].iloc[0]} in this bootstrap sample — skipping")
        
    y_true = (df['ACC'] <= acc_percentile).astype(int).values
    pos_ratio = y_true.mean()
    neg_ratio = 1 - pos_ratio
    print(f"[BOOTSTRAP] Positive class (Get Wrong) ratio: {pos_ratio:.3f}, Negative class(Get Correct) ratio: {neg_ratio:.3f}")



    metrics = []
    for col in df.columns:
        if col == 'ACC':
            continue
        raw = df[col].astype(float).values
        # AUROC
        fpr, tpr, _ = roc_curve(y_true, raw)
        auroc = auc(fpr, tpr)
        # AUPR
        prec, rec, _ = precision_recall_curve(y_true, raw)
        aupr = auc(rec, prec)
        metrics.append([auroc, aupr])
    return np.array(metrics)  # shape: (n_metrics, 2)


def compute_and_save_cis(
    all_metrics: np.ndarray,
    columns: list,
    metric_names: list,
    save_dir: str,
    percentile: str
):
    """
    Compute mean and 95% CI for each metric across bootstrap samples and save CSVs.
    all_metrics shape: (n_bootstrap, n_metrics, 2).
    metric_names: ['AUROC', 'AUPR']
    """
    for idx, mname in enumerate(metric_names):
        # collect shape (n_boot, n_metric)
        mat = all_metrics[:, :, idx]
        mean = mat.mean(axis=0)
        lower = np.percentile(mat, 2.5, axis=0)
        upper = np.percentile(mat, 97.5, axis=0)
        df_out = pd.DataFrame(
            {'mean': mean, 'pctl2.5': lower, 'pctl97.5': upper}, 
            index=columns
        )
        path = os.path.join(save_dir, f'{mname}_{percentile}_bootstrap_ci.csv')
        df_out.to_csv(path)
        logging.info(f"[INFO] Saved {mname} CI to {path}")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--etree_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--acc_percentile', type=float, default=0.75)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.info("[INFO] Loading tree caches...")
    et_cache = general_utils.create_tree_cache(args.etree_dir)
    

    metrics_list = []  # will be list of arrays shape (n_metrics,3)
    logging.info(f"[INFO] Starting {args.n_bootstrap} bootstrap runs...")
    for _ in tqdm(range(args.n_bootstrap), desc="Bootstrapping"):
        sample_df = bootstrap_pairs(et_cache)
        # ipdb.set_trace()
        # sample_df = sample_df.dropna()
        arr = compute_bootstrap_metrics(sample_df, args.acc_percentile)
        metrics_list.append(arr)

    all_metrics = np.stack(metrics_list, axis=0)
    metric_names = ['AUROC', 'AUPR']
    cols = [c for c in sample_df.columns if c != 'ACC']

    logging.info("[INFO] Computing confidence intervals for metrics...")
    compute_and_save_cis(all_metrics, cols, metric_names, args.save_dir, args.acc_percentile)
    logging.info("[INFO] All bootstrap CIs saved.")

if __name__ == '__main__':
    main()
