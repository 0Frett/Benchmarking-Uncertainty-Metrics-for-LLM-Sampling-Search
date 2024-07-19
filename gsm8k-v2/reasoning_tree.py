from collections import defaultdict
from typing import Generic, NamedTuple, List, Dict, Tuple, Callable, Any, Union, Optional, TypedDict
import numpy as np
import warnings
import math
import random
from copy import deepcopy
import itertools
from tqdm import tqdm

class SubResult(TypedDict):
    sub_question: str
    sub_answer: str

class Single_Question_Node:
    def __init__(
        self, 
        question:str,
        state_trace:list[SubResult],
        children:list['Single_Answer_Node'] = None,
    ):
        self.question = question
        self.children = children if children is not None else []
        self.state_trace = state_trace
    
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
        state_trace:list[SubResult],
    ):  
        self.parent = parent
        self.answer = answer
        self.tokens_prob = tokens_prob
        self.state_trace = state_trace
        self.is_terminate = None
        self.is_correct = None

class Single_QA_Struct:
    def __init__(
        self, 
        original_q:str,
        state_trace:list[SubResult],
        parent:'Single_QA_Struct'=None, 
        children:list['Single_QA_Struct']=None,
    ):
        self.original_q = original_q
        self.state_trace = state_trace
        self.parent = parent
        self.children = children if children is not None else []
        self.qa_matches = []  # list[Single_Question_Node<->Single_Answer_Node]

    def add_qa_child(
        self, 
        child:'Single_QA_Struct',
    ):
        self.children.append(child)

class ReasoningTree:
    def __init__(
        self,
        tree_depth:int,
        branch_per_node:int,
        utils:Union['GSM8kUtils', 'FactUtils'],
        truth:Union[int, str, float],
    ):
        self.tree_depth = tree_depth
        self.branch_per_node = branch_per_node
        self.utils = utils
        self.truth = truth

    def run_algo(self):
        root_node = Single_QA_Struct(
            original_q=self.utils.question,
            state_trace=[]
        )
        beam = [root_node]

        for depth in range(self.tree_depth + 1):
            next_level_nodes = []
            for node in tqdm(beam, desc=f"level{depth}"):
                # get perturb qs
                perturb_qs = self.utils.get_perturbed_output(
                    input = node.original_q,
                    n_output = 5 # math.ceil(math.pow(self.branch_per_node, depth+1) / len(beam))
                )
                parent_state_trace = node.state_trace.copy()
                node.qa_matches = [
                    Single_Question_Node(
                        question = q, 
                        state_trace = parent_state_trace + [SubResult(sub_question=q, sub_answer="")]
                    ) for q in perturb_qs
                ]

                # sample answers from perturb qs
                for question_node in node.qa_matches:
                    state_trace = question_node.state_trace.copy()
                    next_actions, next_tokens = self.utils.get_actions(
                        state_trace=state_trace, 
                        n_actions=self.branch_per_node
                    )

                    for idx, action in enumerate(next_actions):
                        # state_trace = question_node.state_trace.copy()
                        state_trace[-1]['sub_answer'] = next_actions[idx]
                        childd = Single_Answer_Node(
                            parent=question_node,
                            answer=self.utils.retrieve_answer(next_actions[idx]),
                            tokens_prob=next_tokens[idx],
                            state_trace=state_trace,
                        )
                        childd.is_terminate = self.utils.judge_terminate(childd.state_trace)
                        childd.is_correct = self.utils.judge_answer(
                            childd.answer, 
                            self.truth
                        ) if childd.is_terminate else None
                        
                        question_node.add_answer_child(childd)
                        
                        if not childd.is_terminate:
                            # add to next level beams if not terminate
                            new_qa_struct = Single_QA_Struct(
                                original_q=childd.state_trace[-1]['sub_answer'],
                                state_trace=childd.state_trace,
                                parent=node,
                            )
                            node.add_qa_child(new_qa_struct)
                            next_level_nodes.append(new_qa_struct)

            beam = next_level_nodes
        
        return root_node

