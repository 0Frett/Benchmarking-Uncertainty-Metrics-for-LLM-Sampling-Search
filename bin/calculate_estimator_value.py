import numpy as np
import os
from data_structs import Estimator_Tree
import metric_utils
import general_utils
import gendata_utils
import gc
import math
from tqdm import tqdm
import torch
from typing import List, Union, Any, Optional, Literal
import evaluate
import argparse
import copy
from llm_utils import vlmModel, OpenAIModel_parallel
from encode_utils import openaiEmbedModel, HiddenStateExtractor
import logging
rouge = evaluate.load("rouge")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)


def calculate_npe(root: Estimator_Tree):
    for subtree in root.subtrees:
        subtree_sample_logprobsum = []
        for a_node in subtree.sampling_answers:
            logprobsum = 0
            for token_info in a_node.token_probs:
                top1_logprob = token_info.get('main_logprob', 0)
                logprobsum += top1_logprob
            a_node.metadata["logprob_sum"] = logprobsum
            subtree_sample_logprobsum.append(logprobsum)
        subtree.metadata["NPE"] = -np.mean(subtree_sample_logprobsum)
    all_vals = [st.metadata["NPE"] for st in root.subtrees]
    root.metadata["NPE"] = np.mean(all_vals)
    logging.info(f"NPE: {root.metadata['NPE']}")


def calculate_lnpe(root: Estimator_Tree):
    for subtree in root.subtrees:
        subtree_sample_logprobsum = []
        for a_node in subtree.sampling_answers:
            logprobsum, tokennum = 0, 0
            for token_info in a_node.token_probs:
                top1_logprob = token_info.get('main_logprob', 0)
                logprobsum += top1_logprob
                tokennum += 1
            a_node.metadata["normalized_logprob_sum"] = logprobsum/tokennum
            subtree_sample_logprobsum.append(logprobsum/tokennum)
        subtree.metadata["LNPE"] = -np.mean(subtree_sample_logprobsum)
    all_vals = [st.metadata["LNPE"] for st in root.subtrees]
    root.metadata["LNPE"] = np.mean(all_vals)
    logging.info(f"LNPE: {root.metadata['LNPE']}")


def calculate_se(
    root: Estimator_Tree,
    encoder: Union[openaiEmbedModel, HiddenStateExtractor],
    use_sentiment: bool = True
):
    for subtree in root.subtrees:
        clusters = metric_utils.se_cluster(
            a_nodes = subtree.sampling_answers,
            sentiment = use_sentiment,
            encoder = encoder,
        )
        # Compute the log probability for each semantic cluster using the log-sum-exp trick.
        cluster_log_probs = []
        for nodes in clusters.values():
            # Retrieve each answer's log probability
            node_log_probs = [a_node.metadata["logprob_sum"] for a_node in nodes]
            cluster_lp = metric_utils.log_sum_exp(node_log_probs)
            cluster_log_probs.append(cluster_lp)
        # Monte Carlo semantic entropy: negative average of cluster log probabilities.
        semantic_entropy = -np.mean(cluster_log_probs) if cluster_log_probs else 0.0
        subtree.metadata["SE"] = semantic_entropy
    all_vals = [st.metadata["SE"] for st in root.subtrees]
    root.metadata["SE"] = np.mean(all_vals)
    logging.info(f"SE: {root.metadata['SE']}")


def calculate_lexical_distance(root: Estimator_Tree):
    for subtree in root.subtrees:
        texts = []
        for a_node in subtree.sampling_answers:
            text = []
            for token_info in a_node.token_probs:
                token = token_info.get('main_token', "")
                text.append(token)
            texts.append(text)

        if len(texts) < 2:
            lex_dist = 0
        else:
            dist_sum = 0.0
            count = 0
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    dist_sum += (1 - metric_utils.lexical_distance(texts[i], texts[j], 'rouge-l'))
                    # results = rouge.compute(predictions=[texts[i]], references=[texts[j]], rouge_types=["rougeL"])
                    # dist_sum += (1 - results["rougeL"])
                    count += 1
            lex_dist = dist_sum / count if count > 0 else 0.0
        subtree.metadata["Lexical"] = lex_dist
    all_vals = [st.metadata["Lexical"] for st in root.subtrees]
    root.metadata["Lexical"] = np.mean(all_vals)
    logging.info(f"Lexical: {root.metadata['Lexical']}")


def calculate_ptrue_complement(
    root: Estimator_Tree, 
    llm: Union[vlmModel, OpenAIModel_parallel],
    sample: int = 1
):
    for subtree in root.subtrees:
        question_text = subtree.paraphrased_question.question
        qs = [question_text]*len(subtree.sampling_answers)
        ans = [a_node.raw_answer for a_node in subtree.sampling_answers]
        p_trues = metric_utils.ptrue_batch(qs, ans, llm, sample=sample)
        for a_node, ptrue in zip(subtree.sampling_answers, p_trues):
            a_node.metadata["PTrue_Comp"] = 1-ptrue
        subtree.metadata["PTrue_Comp"] = np.mean(1-np.array(p_trues))
    all_vals = [st.metadata["PTrue_Comp"] for st in root.subtrees]
    root.metadata["PTrue_Comp"] = np.mean(all_vals)
    logging.info(f"PTrue_Comp: {root.metadata['PTrue_Comp']}")


def calculate_vc_neg(
    root: Estimator_Tree, 
    llm: Union[vlmModel, OpenAIModel_parallel],
    sample: int = 1
):
    for subtree in root.subtrees:
        question_text = subtree.paraphrased_question.question
        qs = [question_text]*len(subtree.sampling_answers)
        ans = [a_node.raw_answer for a_node in subtree.sampling_answers]
        vcs = metric_utils.vc_batch(qs, ans, llm, sample=sample)
        for a_node, vc in zip(subtree.sampling_answers, vcs):
            a_node.metadata["VC_Neg"] = -1*vc
        subtree.metadata["VC_Neg"] = -1*np.mean(vcs)
    all_vals = [st.metadata["VC_Neg"] for st in root.subtrees]
    root.metadata["VC_Neg"] = np.mean(all_vals)
    logging.info(f"VC_Neg: {root.metadata['VC_Neg']}")

def gen_spuq_root(
    root: Estimator_Tree, 
    llm: Union[vlmModel, OpenAIModel_parallel], 
    utils: Any,
    sampling_num: int = 1, 
    branch_num: int = 5, 
    choice_text: Optional[str] = None, 
    passage: Optional[str] = None
):
    spuq_root = gendata_utils.generate_estimator_tree(
        root.original_question, root.ground_truth,
        sampling_num, branch_num, llm, utils, choice_text, passage)
    root.metadata["SPUQ_Tree"] = spuq_root
    

def calculate_spuq_complement(
    root: Estimator_Tree, 
    encoder: Union[openaiEmbedModel, HiddenStateExtractor], 
    dist: Literal["cosine", "l1", "l2"] = "cosine",
):
    spuq_root = root.metadata["SPUQ_Tree"]
    qs, ans = [], []
    for subtree in spuq_root.subtrees:
        qs.append(subtree.paraphrased_question.question)
        ans.append(subtree.sampling_answers[0].raw_answer)
    q_eval_pairs = [[root.original_question, q] for q in qs]
    weights = 1 - np.array(metric_utils.semantic_distance(q_eval_pairs, encoder, dist))
    numerator, denominator = 0, 0
    for idx, a in enumerate(ans):
        refs = ans[:idx] + ans[idx+1:]
        a_pairs = [[a, ref_a] for ref_a in refs]
        a_similarity = 1 - np.array(metric_utils.semantic_distance(a_pairs, encoder, dist))
        numerator += np.sum(a_similarity)*weights[idx]
        denominator += len(refs)*weights[idx]
    root.metadata["SPUQ_Comp"] = 1 - (numerator/denominator)
    logging.info(f"SPUQ_Comp: {root.metadata['SPUQ_Comp']}")


def calculate_eu_ipt(
    root: Estimator_Tree, 
    llm: Union[vlmModel, OpenAIModel_parallel],
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    branch_num: int = 32,
    depth: int = 2,
):
    ipt_tree = gendata_utils.generate_ipt_tree(
        root.original_question, root.ground_truth,
        branch_num, depth, llm, utils, choice_text, passage
    )
    trial_seq_matrix = []
    for ipt_chain in ipt_tree.chains:
        answer_seq = [a.extract_answer for a in ipt_chain.answer_node_seq]
        trial_seq_matrix.append(answer_seq)
    eu = metric_utils.ipt_mi_estimate(trial_seq_matrix)
    root.metadata["IPT_EU"] = eu
    root.metadata["IPT_Tree"] = ipt_tree
    logging.info(f"IPT_EU: {root.metadata['IPT_EU']}")
  

def calculate_acc(root: Estimator_Tree):
    for subtree in root.subtrees:
        correctness = [a_node.is_correct for a_node in subtree.sampling_answers]
        subtree.metadata["ACC"] = correctness.count(True)/len(correctness)
    all_vals = [st.metadata["ACC"] for st in root.subtrees]
    root.metadata["ACC"] = np.mean(all_vals)
    logging.info(f"ACC: {root.metadata['ACC']}")



def calculate_ansu(root: Estimator_Tree):
    all_ans = []
    for subtree in root.subtrees:
        for a_node in subtree.sampling_answers:
            all_ans.append(a_node.extract_answer)
    root.metadata["ANSU"] = metric_utils.calculate_entropy(all_ans)
    logging.info(f"ANSU: {root.metadata['ANSU']}")


def calculate_cu(root: Estimator_Tree):
    all_correctness = []
    for subtree in root.subtrees:
        for a_node in subtree.sampling_answers:
            all_correctness.append(a_node.is_correct)
    root.metadata["CU"] = metric_utils.calculate_entropy(all_correctness)
    logging.info(f"CU: {root.metadata['CU']}")


def calculate_ensemble_au_eu(root: Estimator_Tree):
    def split_into_four(lst):
        n = len(lst)
        k, r = divmod(n, 4)
        parts = [lst[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(4)]
        return parts

    def count_single_hypothesis_label_probs(values:List, answer_set:set):
        total = len(values)
        label_prob_dict = {}
        for ans in answer_set:
            label_prob_dict[ans] = values.count(ans)/total
        return label_prob_dict
    
    def ensemble_approx_au_eu(root: Estimator_Tree, answer_set:set):
        entropy_for_each_hypothesis = []
        label_probs_of_each_hypothesis = {ans:[] for ans in answer_set}

        subtrees = split_into_four(root.subtrees[0].sampling_answers)

        for subtree in subtrees:
            h_ans = [a_node.extract_answer for a_node in subtree]
            prob_dict = count_single_hypothesis_label_probs(h_ans, answer_set)
            for key, value in prob_dict.items():
                label_probs_of_each_hypothesis[key].append(value)
            entropy_for_each_hypothesis.append(metric_utils.calculate_entropy(h_ans))
        
        # update node values
        aleatoric_uncertainty = np.mean(entropy_for_each_hypothesis)
        hypotheses_average_label_probs = [
            np.mean(hypotheses_label_prob) for hypotheses_label_prob in label_probs_of_each_hypothesis.values()
        ]
        entropy = 0.0
        for probability in hypotheses_average_label_probs:
            if probability >0:
                entropy -= probability * math.log2(probability)
        answer_total_uncertainty = entropy
        epistemic_uncertainty = answer_total_uncertainty - aleatoric_uncertainty
        return aleatoric_uncertainty, epistemic_uncertainty

    # collect all ans
    all_ans = []
    for subtree in root.subtrees:
        for a_node in subtree.sampling_answers:
            all_ans.append(a_node.extract_answer)
    ans_set = set(all_ans)
    au, eu = ensemble_approx_au_eu(root, ans_set)

    root.metadata["Ensemble_AU"] = au
    root.metadata["Ensemble_EU"] = eu
    logging.info(f"Ensemble_AU: {root.metadata['Ensemble_AU']}")
    logging.info(f"Ensemble_EU: {root.metadata['Ensemble_EU']}")


def run_add_metrics(root: Estimator_Tree):
    if not root.metadata.get('ANSU', False):
        logging.info("[INFO] Calculating ANSU...")
        calculate_ansu(root)
    else:
        logging.info("[SKIP] ANSU already present.")

    if not root.metadata.get('CU', False):
        logging.info("[INFO] Calculating CU...")
        calculate_cu(root)
    else:
        logging.info("[SKIP] CU already present.")

    if not root.metadata.get('Ensemble_AU', False):
        logging.info("[INFO] Calculating Ensemble_AU and Ensemble_EU...")
        calculate_ensemble_au_eu(root)
    else:
        logging.info("[SKIP] Ensemble_AU and Ensemble_EU already present.")


def run_low_resource_metrics(root: Estimator_Tree):
    if not root.metadata.get('NPE', False):
        logging.info("[INFO] Calculating NPE...")
        calculate_npe(root)
    else:
        logging.info("[SKIP] NPE already present.")
    
    if not root.metadata.get('LNPE', False):
        logging.info("[INFO] Calculating LNPE...")
        calculate_lnpe(root)
    else:
        logging.info("[SKIP] LNPE already present.")
    
    if not root.metadata.get('Lexical', False):
        logging.info("[INFO] Calculating Lexical Distance...")
        calculate_lexical_distance(root)
    else:
        logging.info("[SKIP] Lexical Distance already present.")
    
    if not root.metadata.get('ACC', False):
        logging.info("[INFO] Calculating Accuracy...")
        calculate_acc(root)
    else:
        logging.info("[SKIP] Accuracy already present.")

def run_llm_usage_metrics(
    root: Estimator_Tree,
    llm: Union[vlmModel, OpenAIModel_parallel],
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str]
):
    if not root.metadata.get('VC_Neg', False):
        logging.info("[INFO] Calculating Verbalized Confidence (Negative)...")
        calculate_vc_neg(root, llm)
    else:
        logging.info("[SKIP] VC_Neg already present.")
    
    if not root.metadata.get('PTrue_Comp', False):
        logging.info("[INFO] Calculating P(True) Complement...")
        calculate_ptrue_complement(root, llm)
    else:
        logging.info("[SKIP] PTrue_Comp already present.")
    
    if not root.metadata.get('IPT_EU', False):
        logging.info("[INFO] Calculating MI-IPT (EU)...")
        calculate_eu_ipt(root, llm, utils, choice_text, passage)
    else:
        logging.info("[SKIP] IPT_EU already present.")
    
    if not root.metadata.get('SPUQ_Tree', False):
        logging.info("[INFO] Generating SPUQ Tree...")
        gen_spuq_root(root, llm, utils, choice_text=choice_text, passage=passage)
    else:
        logging.info("[SKIP] SPUQ_Tree already present.")

def run_encoder_usage_metrics(
    root: Estimator_Tree,
    encoder: Union[vlmModel, OpenAIModel_parallel],
):
    if not root.metadata.get('SE', False):
        logging.info("[INFO] Calculating Semantic Entropy (SE)...")
        calculate_se(root, encoder)
    else:
        logging.info("[SKIP] SE already present.")
    
    if not root.metadata.get('SPUQ', False):
        logging.info("[INFO] Calculating SPUQ Complement...")
        calculate_spuq_complement(root, encoder)
    else:
        logging.info("[SKIP] SPUQ already present.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--etree_dir', type=str, required=True, help="Directory of Raw Estimator Tree")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save Estimator Tree with estimator calculation")
    parser.add_argument('--model', type=str, required=True, help="Name of the LLM model (e.g., 'gpt-4o-mini')")
    parser.add_argument('--dataset', type=str, required=True, help="Name of Dataset")
    parser.add_argument('--tensor_parallel_size', type=int, help="number of gpus")
    parser.add_argument('--gpu_memory_utilization', type=float, help="per GPU memory utilization")
    parser.add_argument('--max_tokens', type=int, required=True, help="max output tokens")
    parser.add_argument('--temperature', type=float, required=True, help="temperature")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    etree_dir = args.etree_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    logging.info("Load Trees")
    tree_cache = general_utils.create_tree_cache(etree_dir)

    logging.info("RUN ADD Metrics")
    for fp, root in tqdm(tree_cache.items(), desc="Trees"):
        save_fp = os.path.join(save_dir, f'{fp}.pkl')
        # try:
        run_add_metrics(root)
        general_utils.save_tree(root=root, filename=save_fp)
        # except:
        #     logging.error(f"Error processing {fp}")
        #     continue
    logging.info("All trees processed.")


    # util = general_utils.get_util(args.dataset)
    # llm = general_utils.get_llm(
    #     args.model, args.max_tokens, args.temperature, 
    #     args.tensor_parallel_size, args.gpu_memory_utilization)
    
    # logging.info("Load Trees")
    # tree_cache = general_utils.create_tree_cache(etree_dir)
    # # tree_cache = general_utils.create_tree_cache(save_dir)

    # logging.info("RUN LLM Usage Metrics")
    # for fp, root in tqdm(tree_cache.items(), desc="Trees"):
    #     save_fp = os.path.join(save_dir, f'{fp}.pkl')
    #     choice_text = root.metadata.get("choice")
    #     passage = root.metadata.get("passage")
    #     try:
    #         run_llm_usage_metrics(root, llm, util, choice_text, passage)
    #         general_utils.save_tree(root=root, filename=save_fp)
    #     except:
    #         logging.error(f"Error processing {fp}")
    #         continue
    
    # del tree_cache
    # gc.collect()

    # tree_cache = general_utils.create_tree_cache(save_dir)
    # logging.info("RUN Low Resource Metrics")
    # for fp, root in tqdm(tree_cache.items(), desc="Trees"):
    #     save_fp = os.path.join(save_dir, f'{fp}.pkl')
    #     try:
    #         run_low_resource_metrics(root)
    #         general_utils.save_tree(root=root, filename=save_fp)
    #     except:
    #         logging.error(f"Error processing {fp}")
    #         continue
    
    # del tree_cache
    # gc.collect()

    # tree_cache = general_utils.create_tree_cache(save_dir)
    # del llm
    # torch.cuda.empty_cache()

    # logging.info("RUN Encoder Usage Metrics")
    # encoder = general_utils.get_encoder(args.model)
    # for fp, root in tqdm(tree_cache.items(), desc="Trees"):
    #     save_fp = os.path.join(save_dir, f'{fp}.pkl')
    #     try:
    #         run_encoder_usage_metrics(root, encoder)
    #         general_utils.save_tree(root=root, filename=save_fp)
    #     except:
    #         logging.error(f"Error processing {fp}")
    #         continue
    # logging.info("All trees processed.")


if __name__ == "__main__":
    main()
