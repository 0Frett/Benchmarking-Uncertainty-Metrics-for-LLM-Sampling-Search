from collections import defaultdict
from typing import Generic, NamedTuple, List, Dict, Tuple, Callable, Any, Union, Optional, TypedDict
import numpy as np
import warnings
import math
import random
from copy import deepcopy
import itertools
from tqdm import tqdm

class Single_Question_Node:
    def __init__(
        self, 
        question:str,
        children:list['Single_Answer_Node'] = None,
    ):
        self.question = question
        self.children = children if children is not None else []
    
    def add_answer_child(
        self, 
        child:'Single_Answer_Node',
    ):
        self.children.append(child)

class Single_Answer_Node:
    def __init__(
        self, 
        parent:Single_Question_Node, 
        answer:str, 
        tokens_prob:list, 
    ):  
        self.parent = parent
        self.answer = answer
        self.tokens_prob = tokens_prob
        self.is_correct = None

class Single_Step_QA_Struct:
    def __init__(
        self, 
        original_q:str,
    ):
        self.original_q = original_q
        self.qa_matches = []  # list[Single_Question_Node<->Single_Answer_Node]


class ReasoningNode:
    def __init__(
        self,
        sampling_num:int,
        perturb_num:int,
        utils:Union['Single_Step_GSM8kUtils'],
        truth:Union[int, str, float],
    ):
        self.sampling_num = sampling_num
        self.perturb_num = perturb_num
        self.utils = utils
        self.truth = truth

    def run_algo(self):
        node = Single_Step_QA_Struct(original_q=self.utils.question)
        # get perturb qs
        if self.perturb_num > 1:
            qs = self.utils.get_perturbed_output(
                input = node.original_q,
                n_output = self.perturb_num
            )
        else:
            qs = [node.original_q]
        
        node.qa_matches = [Single_Question_Node(question = q) for q in qs]
        # sample answers from perturb qs
        for question_node in node.qa_matches:
            next_actions, next_tokens = self.utils.get_actions(
                question=question_node.question, 
                n_actions=self.sampling_num,
            )
            for idx, action in enumerate(next_actions):
                childd = Single_Answer_Node(
                    parent=question_node,
                    answer=self.utils.retrieve_answer(next_actions[idx]),
                    tokens_prob=next_tokens[idx],
                )
                childd.is_correct = self.utils.judge_answer(childd.answer, self.truth)
                question_node.add_answer_child(childd)
        
        return node
                