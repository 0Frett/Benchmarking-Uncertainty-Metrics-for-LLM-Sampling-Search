import io
import re
import math
from typing import Optional, Union, Dict, List

class Single_Step_StrategyQAUtils():
    def __init__(
        self, 
        prompt_pool:Dict[str, List[str]], 
        language_model:Union['OpenAIModel_parallel', 'LlamaModel'], 
        problem:str
    ):
        # super().__init__(prompt_pool, language_model)
        self.prompt_pool = prompt_pool
        self.language_model = language_model
        self.question = problem

    def retrieve_answer(self, output: str):
        answer = 'NA'
        if 'Yes' in output or 'yes' in output:
            answer = 'True'
        if 'No' in output or 'no' in output:
            answer = 'False'

        print("====retrieve answer======")
        print(answer)
        print("==========")

        return answer

    def get_perturbed_output(self, input:str, n_output:int):
        model_input = self.prompt_pool['paraphrase_prompt'].format(
            text=input, 
            few_shot_examples=self.prompt_pool['paraphrase_examples']
        )
        gen_output = self.language_model.generate(prompt=model_input, num_return_sequences=n_output)
        text_outputs = [output.strip() for output in gen_output.text]
        print("=====get perturb=====")
        print(text_outputs)
        print("==========")
        return text_outputs

    def judge_answer(self, answer:str, truth:str):
        return answer == 'True'

    def get_actions(self, question:str, n_actions:int):
        model_input = f"{self.prompt_pool['few_shot_examples']}\nQ: {question}\nA:"
        gen_output = self.language_model.generate(prompt=model_input, num_return_sequences=n_actions)
        print(gen_output.text)

        return gen_output.text, gen_output.log_prob
