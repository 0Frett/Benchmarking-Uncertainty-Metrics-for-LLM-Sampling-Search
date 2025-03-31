import json
from typing import Generic, List, Dict, Tuple, Callable, Any, Union, Optional
from datasets import load_dataset
import copy
import pickle
import os
from collections import Counter
from llms_parallel import OpenAIModel_parallel, LlamaModel, GemmaModel
from verifiers import OpenAIModel
from qa_struct import Single_Step_QA_Struct, Single_Answer_Node, Single_Question_Node
from tqdm import tqdm
import random
import re
from tqdm import tqdm
import io
import argparse

# eval_llm = OpenAIModel(model="gpt-4o-mini", max_tokens=50, temperature=1.0)
perturb_llm = OpenAIModel(model="gpt-4o-mini", max_tokens=250, temperature=1.2)

def sample_prompt(prompt_pool, num_shot):
    exs = random.sample(prompt_pool['interactive_examples'], num_shot)
    with io.StringIO() as f:
        f.write(prompt_pool['instruction'] + '\n\n')
        for idx, example in enumerate(exs):
            f.write(f"Question {idx+1}: " + example + '\n\n')
        meta_prompt = f.getvalue()  # instruction and few shot examples
    return meta_prompt


def get_perturbed_output(inputq, n_output, prompt_pool, llm):
    model_input = prompt_pool['paraphrase_prompt'].format(text=inputq)
    gen_output = perturb_llm.generate(prompt=model_input, num_return_sequences=n_output)
    text_outputs = gen_output #[output.strip() for output in gen_output.text]
    print(text_outputs)
    return text_outputs


def sampling_output(question:str, sample_num:int, meta_prompt, prompt_pool, llm, n_shots):
    with io.StringIO() as f:
        f.write(meta_prompt)
        f.write(prompt_pool["question_prefix"].format(question=question) + "\n")
        model_input = f.getvalue()
    gen_output = llm.generate(prompt=model_input, num_return_sequences=sample_num)
    print(gen_output.text)
    return gen_output.text, gen_output.log_prob


def parse_ground_truth(text):
    return text

def retrieve_answer(output: str):
    output = output.lower()
    # Extract text after the phrase
    answer_part = output.split("answer to the question is")[-1]
    # Search for a choice (A, B, C, D, or E)
    match = re.search(r'\b([A-E])\b', answer_part, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Return as uppercase
    else:
        return 'Z'  # no valid choice is found


def judge_answer(answer, groundtruth):
    if answer == groundtruth:
        print('Correct')
        return True
    else:
        print('Wrong')
        return False


def main(question, truth, prompt_pool, language_model, sampling_num, perturb_num, n_shots):
    meta_prompt = sample_prompt(prompt_pool, n_shots)
    node = Single_Step_QA_Struct(original_q=question)
    # get perturb qs
    if perturb_num > 1:
        qs = get_perturbed_output(node.original_q, perturb_num, prompt_pool, language_model)
    else:
        qs = [node.original_q]
    
    node.qa_matches = [Single_Question_Node(question=q) for q in qs]
    # sample answers from perturb qs
    for question_node in node.qa_matches:
        next_actions, next_tokens = sampling_output(question_node.question, sampling_num, meta_prompt, prompt_pool, language_model, n_shots)
        for idx, action in enumerate(next_actions):
            childd = Single_Answer_Node(
                parent=question_node,
                answer=retrieve_answer(next_actions[idx]),
                tokens_prob=next_tokens[idx],
            )
            
            childd.is_correct = judge_answer(childd.answer, parse_ground_truth(truth))
            question_node.add_answer_child(childd)

    return node



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Inferernce')
    parser.add_argument('--model', type=str, default='llama',
                        help='Which model to use: "llama" or "gpt-3.5-turbo" (default: "llama")')
    args = parser.parse_args()
    
    sampling_num = 10
    perturb_num = 5
    n_shots = 0
    model_name = args.model

    print(f'sampling_num:{sampling_num}\nperturb_num:{perturb_num}')

    with open('data/prompt_pool.json') as f:
        prompt_pool = json.load(f)
    with open('data/CommonsenseQA2.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if model_name == 'gpt-3.5-turbo':
        llm = OpenAIModel_parallel(model='gpt-3.5-turbo', max_tokens=500, temperature=1.2)
    if model_name == 'llama':
        llm = LlamaModel(max_tokens=500, temperature=0.3)
    if model_name == 'gemma':
        llm = GemmaModel(max_tokens=500, temperature=0.3)

    for idx in tqdm(range(len(dataset['question'])), desc="Processing Questions"):
        question = dataset['question'][idx]
        truth = dataset['answer'][idx]
        node = main(question, truth, prompt_pool, llm, sampling_num, perturb_num, n_shots)

        # save node
        save_dir = f'./output_nodes/{model_name}/Q{idx+101}/node_1'
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/node.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(node, f)




