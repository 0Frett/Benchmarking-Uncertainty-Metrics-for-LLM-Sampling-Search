import json
from typing import Generic, List, Dict, Tuple, Callable, Any, Union, Optional
from datasets import load_dataset
import copy
import pickle
import os
from collections import Counter
from llms import OpenAIModel, LlamaModel
from llms_parallel import OpenAIModel_parallel
from reasoning_tree import ReasoningTree
from utils import GSM8kUtils, FactUtils
from tqdm import tqdm
import random

def select_difficult_question(dataset, topk_hard):
    """select question needing more reasoning step"""
    answer = dataset['answer']
    steps = []
    for a in answer:
        step = len(a.split("\n"))
        steps.append(step)
    count = Counter(steps)
    select_difficulties = list(dict(sorted(count.items())).keys())[-1*topk_hard:]
    indices = [i for i, x in enumerate(steps) if x in select_difficulties]

    return indices

class gsm8k_Inference():
    def __init__(
        self, 
        prompt_pool:Dict[str, List[str]],
        language_model:Union[OpenAIModel, LlamaModel],
        tree_depth:int,
        branch_per_node:int,
    ):
        self.full_dataset = load_dataset("openai/gsm8k", 'main', split='test')
        self.prompt_pool = prompt_pool
        self.language_model = language_model
        self.tree_depth = tree_depth
        self.branch_per_node = branch_per_node

    def sample_prompt(
        self, 
        num_shot:int = 2,
    ):
        ret = copy.deepcopy(self.prompt_pool)
        ret['interactive_examples'], ret['useful_examples'] = zip(
            *random.sample(
                list(zip(ret['interactive_examples'],ret['useful_examples'])),
                k=num_shot
            )
        )
        return ret
    
    def run_inference(
        self,
        start_idx:int, 
        end_idx:int,
    ):

        self.dataset = list(self.full_dataset)[start_idx:end_idx]

        for i, example in enumerate(
            tqdm(
                self.dataset,
                total=start_idx + len(self.dataset),
                initial=start_idx,
                desc=f"gsm8k_tree{start_idx}~{end_idx}"
            )
        ):
            
            tree_utils = GSM8kUtils(
                prompt_pool=self.sample_prompt(),
                language_model=self.language_model,
                tree_depth=self.tree_depth,
                problem=example['question']
            )

            true_answer = tree_utils.retrieve_true_answer(example['answer'])
            print('true-answer:', true_answer)
            tree_root = ReasoningTree(
                tree_depth=tree_depth,
                branch_per_node=self.branch_per_node,
                utils=tree_utils,
                truth=true_answer
            ).run_algo()
            
            # save tree
            os.makedirs('./output_trees', exist_ok=True)
            save_dir = f'./output_trees/Q{start_idx+i}'
            os.makedirs(save_dir, exist_ok=True)
            save_dir += f'/tree_{len(os.listdir(save_dir))+1}'
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{save_dir}/tree.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(tree_root, f)


if __name__ == '__main__':
    # args
    llm_name = 'gpt-3.5-turbo'
    branch_per_node = 5
    tree_depth = 3
    assert tree_depth % 2 == 1
    temperature = 0.5
    print(f'branch_per_node:{branch_per_node}\ntree_depth:{tree_depth}\ntemperature:{temperature}')
    # start_idx = 0
    # end_idx = 1
    q_idx = [2] #, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218
    with open('./prompts/prompt_pool.json') as f:
        prompt_pool = json.load(f)

    llm = OpenAIModel_parallel(
        model=llm_name,
        max_tokens=300,
        temperature=temperature,
    )

    for idx in q_idx:
        gsm8k_Inference(
            prompt_pool=prompt_pool, 
            language_model=llm, 
            tree_depth=tree_depth,
            branch_per_node=branch_per_node
        ).run_inference(start_idx=idx, end_idx=idx+1)
    
    llm.usage()



