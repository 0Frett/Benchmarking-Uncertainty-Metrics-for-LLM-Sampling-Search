import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float

GSM8kState = list[SubResult]

class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class GSM8kWorld:
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self, language_model, n_confidence=5, temperature=0.8):
        self.language_model = language_model
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.prompt_examples = ""
        self.n_shots = 0


    def update_example(self, example, prompt: GSM8kPromptDict):
        self.example = example
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def step(self, state: GSM8kState, action:str):
        state = state.copy()
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1) + " " + action + "\n")
            f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            model_input = f.getvalue()
        #print(f"world step input : {model_input}")
        
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        for idx in range(self.n_confidence):
            gen_output = self.language_model.generate(prompt=model_input, temperature=self.temperature)
            output = utils.refine_subanswer(gen_output.log_prob[0])
            #print(f"world step output : {output}")
            result = output.strip()
            answer = utils.retrieve_answer(result)  
            #print(f"world step answer : {answer}")
            answer_dict[answer].append(result)

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())
        state.append(SubResult(action, answer, confidence))

        return state


    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False
