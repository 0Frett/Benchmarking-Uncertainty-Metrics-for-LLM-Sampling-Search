import pickle
import os
import json
from tqdm import tqdm
import copy
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
from data_structs import GT_Tree, Estimator_Tree
import llm_utils
import encode_utils
from task_utils import gsm8k_utils, math_utils, commonsenseqa_utils, triviaqa_utils, truthfulqa_utils, ambigqa_utils
from transformers import AutoTokenizer

MCQ_UTILS = ["commonsenseQA", "truthfulQA"]
RCQ_UTILS = ["triviaQA"]
ESTIMATOR_KEYS = ["NPE", "LNPE", "SE", "VC_Neg", "PTrue_Comp", "Lexical", "IPT_EU", "SPUQ_Comp", "ACC"]
SOURCE_KEYS = ["EU", "SU", "AU", "OU"]

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_tree(root: Union[Estimator_Tree, GT_Tree], filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(root, f)

def create_test_tree_cache(data_dir: str) -> Dict[str, Union[GT_Tree, Estimator_Tree]]:
    tree_cache = {}
    tree_fps = os.listdir(data_dir)
    for tree_fp in tqdm(tree_fps, desc="Load Tree into RAM"):
        try:
            full_path = os.path.join(data_dir, tree_fp)
            root = load_tree(full_path)
            tree_cache[Path(tree_fp).stem.split("_")[-1]] = copy.deepcopy(root)
        except Exception as e:
            print(f"Error loading {tree_fp}: {e}")
            continue
        break
    return tree_cache

def create_tree_cache(data_dir: str) -> Dict[str, Union[GT_Tree, Estimator_Tree]]:
    tree_cache = {}
    tree_fps = os.listdir(data_dir)
    for tree_fp in tqdm(tree_fps, desc="Load Tree into RAM"):
        try:
            full_path = os.path.join(data_dir, tree_fp)
            root = load_tree(full_path)
            tree_cache[Path(tree_fp).stem.split("_")[-1]] = copy.deepcopy(root)
        except Exception as e:
            print(f"Error loading {tree_fp}: {e}")
            continue
    return tree_cache

def get_llm(
    model_name, 
    max_tokens=512,
    temperature=0.8, 
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5
) -> Union[llm_utils.vlmModel, llm_utils.OpenAIModel_parallel]:

    if model_name in llm_utils.OPENAI_API_MODELS:
        llm = llm_utils.OpenAIModel_parallel(
            model=model_name, max_tokens=max_tokens, temperature=temperature
        )
    elif model_name in llm_utils.VLM_MODELS:
        llm = llm_utils.vlmModel(
            model=model_name, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return llm

def get_encoder(
    model_name,
) -> Union[encode_utils.HiddenStateExtractor, encode_utils.openaiEmbedModel]:
    
    if model_name in encode_utils.OPENAI_API_ENCODER:
        encoder = encode_utils.openaiEmbedModel(model=model_name)
    elif model_name in encode_utils.HF_ENCODER:
        encoder = encode_utils.HiddenStateExtractor(model_name)
    elif model_name in encode_utils.VLM_ENCODER:
        encoder = encode_utils.vlmEmbedModel(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return encoder

def get_util(data_name):
    if data_name == 'gsm8k':
        return gsm8k_utils
    elif data_name == 'math':
        return math_utils
    elif data_name == 'commonsenseQA':
        return commonsenseqa_utils
    elif data_name == 'triviaQA':
        return triviaqa_utils
    elif data_name == 'truthfulQA':
        return truthfulqa_utils
    elif data_name == 'ambigQA':
        return ambigqa_utils
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

def get_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset