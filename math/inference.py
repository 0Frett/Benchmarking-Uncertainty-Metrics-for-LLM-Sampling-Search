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

eval_llm = OpenAIModel(model="gpt-4o-mini", max_tokens=50, temperature=1.0)
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
    # print(text_outputs)
    return text_outputs


def sampling_output(question:str, sample_num:int, meta_prompt, prompt_pool, llm, n_shots):
    with io.StringIO() as f:
        f.write(meta_prompt)
        f.write(prompt_pool["question_prefix"].format(idx=n_shots+1, question=question) + "\n")
        model_input = f.getvalue()
    gen_output = llm.generate(prompt=model_input, num_return_sequences=sample_num)
    return gen_output.text, gen_output.log_prob


def parse_ground_truth(text):
    start = text.find(r'\boxed{')
    if start == -1:
        return None  # No \boxed found
    # Initialize variables
    start += len(r'\boxed{')  # Move to the start of content inside \boxed
    brace_count = 1
    content = []
    # Traverse character by character to handle nested braces
    for char in text[start:]:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        # Stop once all braces are closed
        if brace_count == 0:
            break
        content.append(char)
    return ''.join(content)



def judge_answer(answer, groundtruth):
    """use gpt to eval the output latex string is same as ground truth or not"""
    eval_prompt = f"""
        I am grading math exams, help me determine whether a student's answer matches the correct answer.
        Student Answer : {answer}\n
        Correct Answer : {groundtruth}\n 
        Does the student answer shares equivalence math meaning with correct answer?
        Output Yes or No.
    """
    print(eval_prompt)
    eval_response = eval_llm.generate(prompt=eval_prompt)[0]
    print(eval_response)
    if eval_response.lower() in {'yes', 'true'}:
        return True
    else:
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
                answer=next_actions[idx],
                tokens_prob=next_tokens[idx],
            )
            
            childd.is_correct = judge_answer(childd.answer, parse_ground_truth(truth))
            question_node.add_answer_child(childd)

    return node



if __name__ == '__main__':
    # args
    sampling_num = 10
    perturb_num = 5
    n_shots = 0
    model_name = 'gemma'

    print(f'sampling_num:{sampling_num}\nperturb_num:{perturb_num}')

    with open('data/prompt_pool.json') as f:
        prompt_pool = json.load(f)
    with open('data/MATH_MIX.json', 'r', encoding='utf-8') as f:
        MATH_dataset = json.load(f)

    if model_name == 'gpt-3.5-turbo':
        llm = OpenAIModel_parallel(model='gpt-3.5-turbo', max_tokens=500, temperature=1.2)
    if model_name == 'llama':
        llm = LlamaModel(max_tokens=500, temperature=0.2)
    if model_name == 'gemma':
        llm = GemmaModel(max_tokens=500, temperature=0.2)

    for idx in tqdm(range(len(MATH_dataset['question'][:70])), desc="Processing Questions"):
        question = MATH_dataset['question'][idx]
        truth = MATH_dataset['answer'][idx]
        node = main(question, truth, prompt_pool, llm, sampling_num, perturb_num, n_shots)

        # save node
        save_dir = f'./output_nodes/{model_name}/Q{idx+1}/node_1'
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/node.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(node, f)




