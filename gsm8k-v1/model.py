import io
import re
from typing import TypedDict, Optional
import numpy as np
import utils

class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str

class GSM8kModel:
    def __init__(self,
                 language_model,
                 n_actions,
                 temperature,
                 depth_limit,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 terminate_on_depth_limit=True):

        self.language_model = language_model
        self.example = ''
        self.temperature = temperature
        self.n_actions = n_actions
        self.terminate_on_depth_limit = terminate_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.prompt_examples = ""
        self.n_shots = 0
    
    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        self.prompt = prompt
        self.example = example
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

    def get_actions(self, state):
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            if at_depth_limit := self.terminate_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()
        #print(f"model get action input : {model_input}")
    
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        text_outputs = []
        tokens_prob = []
        for idx in range(n_actions):
            gen_output = self.language_model.generate(model_input, temperature=temperature)
            text, logprob = utils.refine_subquestion(gen_output.log_prob[0])
            text_outputs.append(text)
            tokens_prob.append(logprob)

        text_outputs = [output.strip() for output in text_outputs]
        if at_depth_limit:
            text_outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in text_outputs]
        #print(f'text_outputs:{text_outputs}')

        return text_outputs, tokens_prob
