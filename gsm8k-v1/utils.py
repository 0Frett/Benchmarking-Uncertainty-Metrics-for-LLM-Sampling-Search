import re
from typing import Optional, Union

# def retrieve_answer(output: str):
#     match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
#     if match is None:
#         return None
#     answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
#     if '=' in answer:
#         answer = answer[answer.rindex('=') + 1:]
#     return answer

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

    # if output is None:
    #     return False
    # try:
    #     output = int(output)
    #     answer = int(answer)
    #     return output == answer
    # except ValueError:
    #     pass
    # try:
    #     output = float(output)
    #     answer = float(answer)
    #     return output == answer
    # except ValueError:
    #     pass
    # return output == answer
