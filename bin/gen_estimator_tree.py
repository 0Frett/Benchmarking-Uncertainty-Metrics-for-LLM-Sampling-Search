
import os
import pickle
import argparse
import logging
import gendata_utils
import general_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Name of the LLM model (e.g., 'gpt-4o-mini')")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the JSON dataset file")
    parser.add_argument('--dataset', type=str, required=True, help="Name of Dataset")
    parser.add_argument('--branch_num', type=int, required=True, help="Number of arms (must be an integer)")
    parser.add_argument('--sampling_num', type=int, required=True, help="Sampling per qnode (must be an integer)")
    parser.add_argument('--tensor_parallel_size', type=int, help="number of gpus")
    parser.add_argument('--gpu_memory_utilization', type=float, help="per GPU memory utilization")
    parser.add_argument('--max_tokens', type=int, required=True, help="max output tokens")
    parser.add_argument('--temperature', type=float, required=True, help="temperature")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    llm = general_utils.get_llm(
        args.model, args.max_tokens, args.temperature, 
        args.tensor_parallel_size, args.gpu_memory_utilization)
    data = general_utils.get_data(args.data_path)
    util = general_utils.get_util(args.dataset)
    save_dir = f'./emnlp/estimator_trees/{args.dataset}/{args.model.split("/")[-1]}/b{args.branch_num}q{args.sampling_num}'
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in range(len(data['question'])):
        save_path = os.path.join(save_dir, f"est_tree{idx}.pkl")
        if os.path.exists(save_path):
            continue
        
        question = data['question'][idx]
        truth = data['answer'][idx]
        choice_text = data['choices'][idx] if util.name in general_utils.MCQ_UTILS else None
        passage = data['passage'][idx] if util.name in general_utils.RCQ_UTILS else None
        
        root = gendata_utils.generate_estimator_tree(
            original_question = question, ground_truth_text = truth, 
            sampling_num = args.sampling_num, branch_num = args.branch_num, 
            llm = llm, utils = util, choice_text=choice_text, passage=passage)
        root.metadata['choice'] = choice_text
        root.metadata['passage'] = passage
        # save node
        with open(save_path, 'wb') as f:
            pickle.dump(root, f)


if __name__ == '__main__':
    main()