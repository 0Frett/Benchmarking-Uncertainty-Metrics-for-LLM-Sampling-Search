from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class QuestionNode:
    question: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AnswerNode:
    raw_answer: str
    extract_answer: str
    token_probs: List[dict[str, Any]]
    is_correct: Optional[bool]
    metadata: dict = field(default_factory=dict)


@dataclass
class Chain:
    paraphrased_question: QuestionNode
    clarified_question: QuestionNode
    first_trial_answer: AnswerNode
    self_checked_answer: AnswerNode
    metadata: dict = field(default_factory=dict)


@dataclass
class Subtree:
    paraphrased_question: QuestionNode
    sampling_answers: List[AnswerNode]
    metadata: dict = field(default_factory=dict)


@dataclass
class GT_Tree:
    original_question: str
    ground_truth: str
    chains: List[Chain] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Estimator_Tree:
    original_question: str
    ground_truth: str
    subtrees: List[Subtree] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Iterative_Chain:
    answer_node_seq: List[AnswerNode]
    metadata: dict = field(default_factory=dict)


@dataclass
class MI_IPT_Tree:
    original_question: str
    ground_truth: str
    chains: List[Iterative_Chain] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)