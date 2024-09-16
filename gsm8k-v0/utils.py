import io
import re
import math
from typing import Optional, Union, Dict, List
from abc import ABC, abstractmethod


class Single_Step_GSM8kUtils():
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
        self.n_shots = len(self.prompt_pool['interactive_examples'])
        with io.StringIO() as f:
            f.write(self.prompt_pool['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt_pool['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.meta_prompt = f.getvalue()  # instruction and few shot examples


    def retrieve_answer(self, output: str):
        answer = output.split("The answer to the question is")[-1]
        cleaned_text = re.sub(r'[,$]', '', answer)
        numbers = re.findall(r'-?\d+', cleaned_text)
        answer = "Fail" if len(numbers) == 0 else numbers[0]
        print("====retrieve answer======")
        print(answer)
        print("==========")

        return answer

    def retrieve_true_answer(self, answer: str):
        return re.match(r'[\S\s]*#### (.*)$', answer)[1].replace(',', '').replace(' ', '')

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
        return truth in answer

    def get_actions(self, question:str, n_actions:int,):
        with io.StringIO() as f:
            f.write(self.meta_prompt)
            f.write(self.prompt_pool["question_prefix"].format(idx=self.n_shots+1, question=question) + "\n")
            model_input = f.getvalue()
        gen_output = self.language_model.generate(prompt=model_input, num_return_sequences=n_actions)
        print(gen_output.text)

        return gen_output.text, gen_output.log_prob
