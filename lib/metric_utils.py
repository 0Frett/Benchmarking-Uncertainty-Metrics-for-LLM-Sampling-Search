import numpy as np
from typing import List, Union, Literal, Optional, Dict
from collections import Counter
import math
import re
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, normalize
from llm_utils import vlmModel, OpenAIModel_parallel
from encode_utils import openaiEmbedModel, HiddenStateExtractor
from data_structs import AnswerNode
import prompt_templates
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

def softmax(logits: list[float]) -> np.ndarray:
    logits = np.array(logits)
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def calculate_entropy(values:List[str]) -> float:
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
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def log_sum_exp(log_probs: list[float]) -> float:
    m = max(log_probs)
    return m + math.log(sum(math.exp(lp - m) for lp in log_probs))


def lcs(a: List[str], b: List[str]) -> int:
    """Compute the length of the longest common subsequence between two sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    
    return dp[m][n]


def rouge_l(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    """
    Compute ROUGE-L F1 score between candidate and reference strings.
    """
    
    lcs_len = lcs(cand_tokens, ref_tokens)
    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def ngram(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n(
    cand_tokens: List[str], 
    ref_tokens: List[str], 
    n: int
) -> float:
    
    cand_ngrams = Counter(ngram(cand_tokens, n))
    ref_ngrams = Counter(ngram(ref_tokens, n))
    overlap = sum((cand_ngrams & ref_ngrams).values())
    total_cand = sum(cand_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    if total_cand == 0 or total_ref == 0:
        return 0.0

    precision = overlap / total_cand
    recall = overlap / total_ref
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def bleu_n(
    cand_tokens: List[str], 
    ref_tokens: List[str],
    n: int = 4,
) -> float:
    """
    Compute sentence‐level BLEU score between a single candidate and reference.
    """
    # smoothing to avoid zero scores when any n‑gram is missing:
    weights = tuple([1/n] * n)
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothie)


def lexical_distance(
    cand_tokens: List[str], 
    ref_tokens: List[str], 
    metric: Literal["rouge-l", "rouge-n", "bleu-n"],
    n: Optional[int] = 4
) -> float:
    
    if metric == "rouge-l":
        return 1 - rouge_l(cand_tokens, ref_tokens)
    elif metric == "bleu-n":
        return 1- bleu_n(cand_tokens, ref_tokens, n)
    elif metric == "rouge-n":
        return 1- rouge_n(cand_tokens, ref_tokens, n)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def mutual_info_matrix_knn(
    X: np.ndarray,
    n_neighbors: int = 3,
) -> np.ndarray:
    """
    Compute a symmetric MI matrix using sklearn's k-NN estimator.
    
    M[i, j] ≈ I(X[:, i]; X[:, j]) in nats.
    """
    if X.ndim != 2:
        raise ValueError("Input must be 2D: (n_samples, n_features).")
    
    n_features = X.shape[1]
    M = np.zeros((n_features, n_features))
    
    # For each target column j, estimate MI(feature_i, target_j) for all i at once:
    for j in range(n_features):
        # mutual_info_regression(X, y) returns an array of MI(feature_i; y)
        mi_col = mutual_info_regression(
            X,                     # all features as predictors
            X[:, j],               # the j-th column as the “target”
            n_neighbors=n_neighbors
        )
        M[:, j] = mi_col
        M[j, :] = mi_col  # symmetry

    return M

def mutual_information(x: List[float], y: List[float]) -> float:
    """
    Estimate I(X;Y) by discretizing into a 2D histogram.
    Returns MI in nats.
    """
    def freedman_diaconis_bins(u: np.ndarray):
        # u = one-dimensional data array
        N = len(u)
        iqr = np.percentile(u, 75) - np.percentile(u, 25)
        h = 2 * iqr / (N ** (1/3))  # avoid zero
        if h <= 0:
            h = 1e-6
        return int(np.ceil((u.max() - u.min()) / h))
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")

    # Normalize independently
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    x_norm = scaler_x.fit_transform(x.reshape(-1,1)).ravel()
    y_norm = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
    
    Bx = max(1, freedman_diaconis_bins(x_norm))
    By = max(1, freedman_diaconis_bins(y_norm))
    c_xy, _, _ = np.histogram2d(x_norm, y_norm, bins=(Bx, By))
    p_xy = c_xy / c_xy.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Compute MI via vectorized mask
    px_py = np.outer(p_x, p_y)
    mask = p_xy > 0
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / px_py[mask]))

    return mi


def mutual_info_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the mutual information matrix between columns of X using histogram-based estimation.
    Returns a symmetric matrix M where M[i,j] = I(X[:,i]; X[:,j]) in nats.
    """
    # Make sure X is 2D
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array (n_samples, n_features)")

    n_features = X.shape[1]
    M = np.zeros((n_features, n_features))
    # Compute upper triangle (including diagonal)
    for i in range(n_features):
        for j in range(i, n_features):
            mi = mutual_information(X[:, i], X[:, j])
            M[i, j] = M[j, i] = mi  # symmetric
    return M


def js_divergence(x: List[str], y: List[str]) -> float:
    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    def kl_divergence(p, q):
        return sum(p[k] * math.log(p[k] / q[k]) for k in p if p[k] > 0 and q[k] > 0)
    
    # Count frequencies
    p_counts = Counter(x)
    q_counts = Counter(y)
    vocab = set(p_counts) | set(q_counts)
    # Normalize to get probability distributions
    p = normalize({k: p_counts.get(k, 0) for k in vocab})
    q = normalize({k: q_counts.get(k, 0) for k in vocab})
    # Mixture distribution
    m = {k: 0.5 * (p[k] + q[k]) for k in vocab}
    # Compute JS divergence
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def semantic_distance(
    pairs: List[List[str]], 
    encoder: Union[openaiEmbedModel, HiddenStateExtractor], 
    dist: Literal["cosine", "l1", "l2"],
) -> List[float]:

    """
    Compute cosine, L2 (euclidean) and L1 (manhattan) distance for each (a, b) pair.
    """
    batch_size = encoder.max_batch_size
    distances = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        texts: List[str] = []
        for a, b in batch:
            texts.extend([a, b])  # flatten to [a0, b0, a1, b1, ...]
        embs = np.asarray(encoder.process_batch(texts))  # shape = (2*B, dim)
        B = len(batch)

        for j in range(B):
            e1 = embs[2*j].reshape(1, -1)
            e2 = embs[2*j + 1].reshape(1, -1)

            if dist == "cosine":
                e1_norm = normalize(e1, norm='l2')
                e2_norm = normalize(e2, norm='l2')
                distances.append(float(cosine_distances(e1_norm, e2_norm)))
            if dist == "l2":
                distances.append(float(euclidean_distances(e1, e2)))
            if dist == "l1":
                distances.append(float(manhattan_distances(e1, e2)))
    return distances


def ptrue_batch(
    question_text: List[str], 
    raw_answer: List[str], 
    llm: Union[vlmModel, OpenAIModel_parallel],
    sample: int,
    batch_size: int = 32
) -> List[float]:
    
    prompts = [
        prompt_templates.ptrue_prompt.format(q=q, a=a)
        for q, a in zip(question_text, raw_answer)
    ]
    ptrue_mean_values = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_ptrue = []
        for _ in range(sample):
            gens = llm.batch_generate(prompts=batch, num_return_sequences=1)
            ptrues = []
            for g in gens:
                ptrue = 0.0
                # logging.info(f"extract ptrue from text: {g.text[0]}")
                k = 0
                for idx, tok in enumerate(reversed(g.log_prob[0])):
                    if tok['main_token'].lower() in ["true", "false"]:
                        k = 1
                        topk_tokens = [t.lower() for t in tok['topk_tokens']]
                        true_index = topk_tokens.index("true") if "true" in topk_tokens else -1
                        ptrue = np.exp(tok['topk_logprobs'][true_index]) if true_index >= 0 else 0.0
                        # logging.info(f"ptrue value {ptrue} at top {true_index} of reverse index {idx} token")
                        logging.info(f"[REVERSE] Found token at index {idx} position top {true_index}, p(true)={ptrue}")
                        break

                if k == 0:
                    logging.info(f"Fail extract ptrue")
                ptrues.append(ptrue)
            batch_ptrue.append(ptrues)
        ptrue_mean_values += list(np.mean(batch_ptrue, axis=0))
    return ptrue_mean_values


def vc_batch(
    question_text: List[str], 
    raw_answer: List[str], 
    llm: Union[vlmModel, OpenAIModel_parallel],
    sample: int,
    batch_size: int = 32
) -> List[float]:
    
    prompts = [
        prompt_templates.verbC_prompt.format(q=q, a=a)
        for q, a in zip(question_text, raw_answer)
    ]
    verbc_mean_values = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_verbc = []
        for _ in range(sample):
            gens = llm.batch_generate(prompts=batch, num_return_sequences=1)
            verbcs = []
            for g in gens:
                text = g.text[0]
                # logging.info(f"{text}")
                text = text.lower().split("confidence score")[-1]
                # logging.info(f"extract confidence from text: {text}")
                match = re.search(r"boxed\{(\d+)", text)
                conf = int(match.group(1)) if match else 0
                logging.info(f"extract confidence: {conf}")
                verbcs.append(conf)
            batch_verbc.append(verbcs)
        verbc_mean_values += list(np.mean(batch_verbc, axis=0))
    return verbc_mean_values


def se_cluster(
    a_nodes: List[AnswerNode],
    sentiment: bool,
    encoder: Union[openaiEmbedModel, HiddenStateExtractor],
    dist: str = "cosine",
    threshold: float = 0.3,
) -> Dict[str, List[AnswerNode]]:

    if not sentiment:
        # Basic clustering by string match
        clusters = defaultdict(list)
        for a_node in a_nodes:
            key = a_node.extract_answer
            clusters[key].append(a_node)
        return clusters

    # Sentiment-aware clustering using semantic similarity
    clusters: List[List[AnswerNode]] = []

    for a_node in a_nodes:
        inserted = False
        for cluster in clusters:
            # Use the first element as the representative
            pairs = [[cluster[0].raw_answer, a_node.raw_answer]]
            distances = semantic_distance(pairs, encoder, dist)
            if all(d < threshold for d in distances):
                cluster.append(a_node)
                inserted = True
                break
        if not inserted:
            clusters.append([a_node])

    return {str(i): cluster for i, cluster in enumerate(clusters)}


def ipt_mi_estimate(matrix:List[List[str]]):
    def empirical_joint_and_marginals(matrix):
        """
        matrix: List of rows, each row is a tuple/list of discrete values.
        Returns:
        P_joint: dict mapping tuple -> probability
        P_margs: list of dicts, one per column, mapping value -> probability
        """
        n = len(matrix)
        # 1) joint counts
        joint_counts = Counter(tuple(row) for row in matrix)
        P_joint = {x: cnt / n for x, cnt in joint_counts.items()}

        # 2) marginal counts per column
        cols = len(matrix[0])
        marg_counts = [Counter() for _ in range(cols)]
        for row in matrix:
            for i, v in enumerate(row):
                marg_counts[i][v] += 1
        P_margs = [{val: cnt / n for val, cnt in col_cnt.items()} 
                for col_cnt in marg_counts]

        return P_joint, P_margs

    def product_of_marginals(P_margs):
        """
        Given a list of marginal dicts, return Q(x1,…,xk) = ∏ P_margs[i][xi]
        for every possible tuple x = (x1,…,xk) in the *joint support*.
        """
        # to know support, we'll ask the caller to pass joint-support keys.
        def Q(x_tuple):
            p = 1.0
            for i, xi in enumerate(x_tuple):
                p *= P_margs[i].get(xi, 0.0)
            return p
        return Q

    def kl_divergence(P_joint, Q):
        """
        D_KL(P || Q) = Σ_x P(x) * log( P(x) / Q(x) )
        (we skip terms where P(x)=0; if Q(x)=0 but P(x)>0, the KL is infinite)
        """
        kl = 0.0
        for x, p in P_joint.items():
            q = Q(x)
            if p == 0:
                continue
            if q == 0:
                return float('inf')
            kl += p * math.log(p / q)
        return kl
    
    P_joint, P_margs = empirical_joint_and_marginals(matrix)
    q = product_of_marginals(P_margs)
    kl = kl_divergence(P_joint, q)

    return kl