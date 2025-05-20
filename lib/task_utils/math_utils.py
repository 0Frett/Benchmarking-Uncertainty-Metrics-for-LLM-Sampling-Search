import re
import logging
import math_equivalence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

name = "math"

paraphrase_prompt = """
    Paraphrase the following question, without changing its meaning.
    Make sure you only output a single question only.
    Question: {q}
    Paraphrased Question:
"""

clarify_prompt = """
    Clarify the following question by rewriting it in a clearer, more complete form.
    If the question is ambiguous, add missing details to make it understandable.
    Make sure you only output a single question only.    
    Original Question: {q}
    Clarified Question:
"""

# sampling_prompt = """
#     Please answer the following question. 
#     Think carefully and in a step-by-step fashion. 
#     At the end of your solution, box only the final answer (eg. number, expression). 
#     Use LaTeX format like $\\boxed{{your\_answer\_here}}$
#     Q: {q} 
# """

sampling_prompt = """
    Please answer the following question. 
    Think carefully and in a step-by-step fashion. 
    At the end of your solution, put your final result in a boxed environment, 
    e.g. $\\boxed{{answer}}$.
    Q: {q} 
"""

check_prompt = """
    Following is your previous response to the question.
    Q: {q}
    Your previous response: {a}
    Check your previous response carefully and solve the same question again step by step.
    At the end of your solution, put your final result in a boxed environment, 
    e.g. $\\boxed{{answer}}$.
    Output:
"""

iterative_prompt = """
    Consider the following question:
    Question: {q}
    Below are other proposed solutions:
    {aseq}
    
    Now, provide your own solution to the question. 
    At the end of your solution, put your final result in a boxed environment, e.g. $\\boxed{{answer}}$.
"""

def extract_answer(text):
    # text = output.split("answer to the question is")[-1]
    pattern = r"boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).split("=")[-1]
    else:
        # if not match return subsequence
        answer = ""
    return answer


def eval_answer(model_output, ground_truth_text, question):
    ground_truth = extract_ground_truth(ground_truth_text)
    model_pred = extract_answer(model_output)
    logging.info(model_pred)
    logging.info(ground_truth)
    result = math_equivalence.is_equiv(model_pred, ground_truth, verbose=True)
    logging.info(f"eval answer : {result}")
    return result


def extract_ground_truth(text):
    pattern = r"boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
    return answer
