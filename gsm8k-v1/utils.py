import re
from typing import Optional, Union


def retrieve_answer(output: str):
    answer = output.split("The answer is")[-1]
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 0:
        answer = "Fail"
    else:
        answer = numbers[0]
        
    return answer

def refine_subquestion(action_tokens: list):
    """extract first subquestion"""
    text_action = ""
    new_action_tokens = []
    for token_obj in action_tokens:
        text_action += token_obj['token']
        new_action_tokens.append(token_obj)
        if '?' in token_obj['token']:
            break
    
    return text_action, new_action_tokens

def refine_subanswer(action_tokens: list):
    text_answer = ""
    for token_obj in action_tokens:
        if 'Question' in token_obj['token']:
            break
        text_answer += token_obj['token']
    
    return text_answer

def retrieve_true_answer(answer: str):
    return re.match(r'[\S\s]*#### (.*)$', answer)[1].replace(',', '').replace(' ', '')


def judge_answer(output: str, answer: str):
    if answer in output:
        return True
    else:
        return False
