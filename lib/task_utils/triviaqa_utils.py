import re
import logging
from typing import List
import evaluate

rouge = evaluate.load("rouge")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

name = "triviaQA"

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

sampling_prompt = """
    Read the following passage and answer the question. 
    Passage : {p}
    Question : {q} 
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

check_prompt = """
    Following is your previous response to the question:
    Read the following passage and answer the question. 
    Passage : {p}
    Question : {q} 

    Your previous response: {a}
    Check your previous response carefully and respond the question again.
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

iterative_prompt = """
    Consider the following question:
    Passage : {p}
    Question: {q}

    Below are other proposed solutions:
    {aseq}
    
    Now, provide your own solution to the question.  
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

def extract_answer(text):
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()
    return ""

def extract_ground_truth(text) -> List[str]:
    return text.split(",")

def eval_answer(model_output, ground_truth_text, question):
    ground_truths = extract_ground_truth(ground_truth_text)
    model_pred = extract_answer(model_output).lower()
    logging.info(model_pred)
    logging.info(ground_truths)
    results = rouge.compute(
        predictions=[model_pred.lower()], 
        references=[ground_truths], 
        rouge_types=["rougeL"]
    )
    result = True if results["rougeL"] > 0.5 else False
    logging.info(f"rouge-L : {results['rougeL']}")
    logging.info(f"eval answer : {result}")
    return result

