import json
from typing import Generic, List, Dict, Tuple, Callable, Any, Union, Optional
from datasets import load_dataset
import copy
import pickle
import os
from collections import Counter
from llms_parallel import OpenAIModel_parallel
from qa_struct import ReasoningNode
from utils import Single_Step_GSM8kUtils, Single_Step_FactUtils
from tqdm import tqdm
import random

class Single_Step_gsm8k_Inference():
    def __init__(
        self, 
        prompt_pool:Dict[str, List[str]],
        language_model:Union[OpenAIModel_parallel],
        sampling_num:int,
        perturb_num:int,
    ):
        self.full_dataset = load_dataset("openai/gsm8k", 'main', split='test')
        self.prompt_pool = prompt_pool
        self.language_model = language_model
        self.sampling_num = sampling_num
        self.perturb_num = perturb_num

    def sample_prompt(self, num_shot:int = 4):
        ret = copy.deepcopy(self.prompt_pool)
        ret['interactive_examples'], ret['useful_examples'] = zip(
            *random.sample(
                list(zip(ret['interactive_examples'],ret['useful_examples'])),
                k=num_shot
            )
        )
        return ret
    
    def run_inference(self, start_idx:int, end_idx:int):
        print("data num", len(list(self.full_dataset)))
        self.dataset = list(self.full_dataset)[start_idx:end_idx]
        for i, example in enumerate(
            tqdm(
                self.dataset,
                total=start_idx + len(self.dataset),
                initial=start_idx,
                desc=f"gsm8k_q{start_idx}~{end_idx}"
            )
        ):
            
            node_utils = Single_Step_GSM8kUtils(
                prompt_pool=self.sample_prompt(),
                language_model=self.language_model,
                problem=example['question']
            )

            true_answer = node_utils.retrieve_true_answer(example['answer'])
            print('true-answer:', true_answer)
            
            node = ReasoningNode(
                sampling_num=self.sampling_num,
                perturb_num=self.perturb_num,
                utils=node_utils,
                truth=true_answer
            ).run_algo()
            
            # save node
            os.makedirs('./output_nodes', exist_ok=True)
            save_dir = f'./output_nodes/Q{start_idx+i}'
            os.makedirs(save_dir, exist_ok=True)
            save_dir += f'/node_{len(os.listdir(save_dir))+1}'
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{save_dir}/node.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(node, f)


if __name__ == '__main__':
    # args
    llm_name = 'gpt-3.5-turbo'
    sampling_num = 10
    perturb_num = 5
    temperature = 0.7
    print(f'sampling_num:{sampling_num}\nperturb_num:{perturb_num}\ntemperature:{temperature}')
    q_idx = [i for i in range(500, 600)] #, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218
    with open('./prompts/prompt_pool.json') as f:
        prompt_pool = json.load(f)

    llm = OpenAIModel_parallel(
        model=llm_name,
        max_tokens=500,
        temperature=temperature,
    )

    for idx in q_idx:
        Single_Step_gsm8k_Inference(
            prompt_pool=prompt_pool, 
            language_model=llm, 
            sampling_num=sampling_num,
            perturb_num=perturb_num
        ).run_inference(start_idx=idx, end_idx=idx+1)
    
    llm.usage()



