import json
from typing import Generic, List, Dict, Tuple, Callable, Any, Union, Optional
from datasets import load_dataset
import copy
import pickle
import os
from collections import Counter
from llms_parallel import OpenAIModel_parallel, LlamaModel, GemmaModel
from qa_struct import ReasoningNode
from utils import Single_Step_GSM8kUtils
from tqdm import tqdm
import random
import re
#  from hard_qidxs import *

class Single_Step_gsm8k_Inference():
    def __init__(
        self, 
        prompt_pool:Dict[str, List[str]],
        language_model:Union[OpenAIModel_parallel, LlamaModel],
        model_name:str,
        sampling_num:int,
        perturb_num:int,
    ):
        self.full_dataset = load_dataset("openai/gsm8k", 'main', split='test')
        self.prompt_pool = prompt_pool
        self.language_model = language_model
        self.sampling_num = sampling_num
        self.perturb_num = perturb_num
        self.model_name = model_name

    def sample_prompt(self, num_shot:int = 1):
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
            os.makedirs(f'./output_nodes/{self.model_name}', exist_ok=True)
            save_dir = f'./output_nodes/{self.model_name}/Q{start_idx+i}'
            os.makedirs(save_dir, exist_ok=True)
            save_dir += f'/node_{len(os.listdir(save_dir))+1}'
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{save_dir}/node.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(node, f)


if __name__ == '__main__':
    # args
    sampling_num = 10
    perturb_num = 5
    temperature = 0.2
    modelname = 'gemma'
    print(f'sampling_num:{sampling_num}\nperturb_num:{perturb_num}\ntemperature:{temperature}')
    # total_q_idx = [int(re.search(r'\d+', s).group()) for s in os.listdir('./output_nodes/gpt-3.5-turbo/')]
    # have_idx = [int(re.search(r'\d+', s).group()) for s in os.listdir('./output_nodes/llama/')]
    llama = os.listdir("output_nodes/llama")
    gpt = os.listdir("output_nodes/gpt-3.5-turbo")
    gemma = os.listdir("output_nodes/gemma")
    intersection = list((set(llama) & set(gpt)) - set(gemma))
    ss = [int(re.search(r'\d+', s).group()) for s in intersection]
    q_idx = random.sample(ss, 50)
   
    print(q_idx)
    # [i for i in range(500, 600)] #, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218
    with open('./prompts/prompt_pool.json') as f:
        prompt_pool = json.load(f)

    if modelname == 'gpt-3.5-turbo':
        llm = OpenAIModel_parallel(
            model='gpt-3.5-turbo',
            max_tokens=500,
            temperature=temperature,
        )
    if modelname == 'llama':
        llm = LlamaModel(
            max_tokens=200,
            temperature=temperature,
        )
    if modelname == 'gemma':
        llm = GemmaModel(
            max_tokens=512,
            temperature=temperature,
        )

    for idx in tqdm(q_idx, desc="Processing trees"):
        Single_Step_gsm8k_Inference(
            prompt_pool=prompt_pool, 
            language_model=llm,
            model_name=modelname,
            sampling_num=sampling_num,
            perturb_num=perturb_num
        ).run_inference(start_idx=idx, end_idx=idx+1)
    
    # llm.usage()



