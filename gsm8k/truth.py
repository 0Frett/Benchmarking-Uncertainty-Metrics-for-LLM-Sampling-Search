from typing import Generic
from collections import defaultdict
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy
import itertools
from tqdm import tqdm
import utils

class TruthTreeNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state, action:str, action_tokens:List = None, parent= None, children:List = None):
        self.id = next(TruthTreeNode.id_iter)
        self.state = state
        self.action = action
        self.action_tokens = action_tokens
        self.parent = parent
        self.children = children if children is not None else []
        self.leaves = []
        self.correct = False

    def add_child(self, child):
        self.children.append(child)

    def get_trace(self):
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path


class TruthTree:
    def __init__(self, tree_depth, temperature, model, world):
        self.tree_depth = tree_depth
        self.temperature = temperature
        self.model = model
        self.world = world
        #self.tree_nodes = []

    def __call__(self, true_answer):
        TruthTreeNode.reset_id()
        init_state = self.world.init_state() ###\
        root_node = TruthTreeNode(state=init_state, action=None)
        beam = [root_node]
        #self.tree_nodes.append([root_node])

        for depth in range(self.tree_depth+1):
            same_level_nodes = []
            for node in tqdm(beam, desc=f"level{depth}"):
                current_state = node.state
                if self.world.is_terminal(current_state):
                    node.answer = utils.retrieve_answer(current_state[-1].sub_answer)
                    if utils.judge_answer(node.answer, true_answer):
                        node.correct = True
                    continue

                next_actions, action_tokens = self.model.get_actions(state=current_state)

                next_states = []
                for naction in next_actions:
                    next_states.append(self.world.step(state=current_state, action=naction))

                assert len(next_actions) == len(next_states)

                for idx in range(len(next_states)):
                    candidate_node = TruthTreeNode(state=next_states[idx], action=next_actions[idx], action_tokens=action_tokens[idx], parent=node)
                    node.add_child(candidate_node)
                    same_level_nodes.append(candidate_node)
                
            #self.tree_nodes.append(same_level_nodes)
            beam = same_level_nodes

        return root_node
            


