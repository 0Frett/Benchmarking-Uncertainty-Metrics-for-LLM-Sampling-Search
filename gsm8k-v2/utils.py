import io
import re
import math
from typing import Optional, Union, Dict, List
from abc import ABC, abstractmethod

class UtilsBase(ABC):
    def __init__(self, prompt_pool:Dict[str, List[str]], language_model:Union['OpenAIModel', 'LlamaModel']):
        self.prompt_pool = prompt_pool
        self.language_model = language_model
    
    @abstractmethod
    def judge_terminate(self, state_trace:List['SubResult']):
        ...

    @abstractmethod
    def judge_answer(self, answer:Union[str, int], truth:Union[str, int]):
        ...

    @abstractmethod
    def retrieve_answer(self, output:str):
        ...

    @abstractmethod
    def retrieve_true_answer(self, answer:str):
        ...

    @abstractmethod
    def get_perturbed_output(self, input:str, n_output:int):
        ...




class GSM8kUtils(UtilsBase):
    def __init__(self, prompt_pool:Dict[str, List[str]], language_model:Union['OpenAIModel', 'LlamaModel'], tree_depth:int, problem:str):
        super().__init__(prompt_pool, language_model)
        self.depth_limit = tree_depth
        self.question = problem
        self.n_shots = len(self.prompt_pool['interactive_examples'])
        with io.StringIO() as f:
            f.write(self.prompt_pool['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt_pool['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.meta_prompt = f.getvalue()  # instruction and few shot examples

    def retrieve_answer(self, output: str):
        answer = output.split("The answer is")[-1]
        numbers = re.findall(r'\d+', answer)
        if len(numbers) == 0:
            answer = "Fail"
        else:
            answer = numbers[0]
            
        return answer

    def extract_first_subquestion(self, action_tokens: list):
        text_action = ""
        new_action_tokens = []
        for token_obj in action_tokens:
            text_action += token_obj['token']
            new_action_tokens.append(token_obj)
            if '?' in token_obj['token']:
                break
        
        return text_action, new_action_tokens

    def extract_first_subanswer(self, action_tokens: list):
        text_answer = ""
        new_action_tokens = []
        for token_obj in action_tokens:
            if 'Question' in token_obj['token']:
                break
            text_answer += token_obj['token']
            new_action_tokens.append(token_obj)
        
        return text_answer, new_action_tokens

    def retrieve_true_answer(self, answer: str):
        return re.match(r'[\S\s]*#### (.*)$', answer)[1].replace(',', '').replace(' ', '')

    def get_perturbed_output(self, input:str, n_output:int):
        model_input = self.prompt_pool['paraphrase_prompt'].format(text=input, few_shot_examples=self.prompt_pool['paraphrase_examples'])
        gen_output = self.language_model.generate(prompt=model_input, num_return_sequences=n_output)
        print(gen_output.text)
        text_outputs = []
        for output in gen_output.log_prob:
            text, logprob = self.extract_first_subanswer(output)
            text_outputs.append(text)
        text_outputs = [output.strip() for output in gen_output.text]

        return text_outputs

    def judge_terminate(self, state_trace:List['SubResult']):
        if len(state_trace) >= 2:
            if "Now we can answer" in state_trace[-2]['sub_answer']:
                return True
        return False

    def judge_answer(self, answer:str, truth:str):
        return truth in answer

    def get_actions(
        self, 
        state_trace:List['SubResult'],
        n_actions:int,
    ):
        state_trace = state_trace.copy()
        print(state_trace)
        with io.StringIO() as f:
            f.write(self.meta_prompt)
            last_idx = 0
            for idx, subresult in enumerate(state_trace):
                if idx == 0:
                    f.write(self.prompt_pool["question_prefix"].format(idx=self.n_shots+1, question=subresult['sub_question']) + "\n")
                    continue
                if idx % 2 == 0:
                    # even index action : generate question
                    f.write(self.prompt_pool["answer_prefix"].format(idx=self.n_shots+1, sub_idx=math.ceil(idx/2)) + " " + subresult['sub_question'] + "\n")
                else:
                    # odd index action : generate answer
                    f.write(self.prompt_pool["subquestion_prefix"].format(idx=self.n_shots+1, sub_idx=math.ceil(idx/2)) + " " + subresult['sub_question'] + "\n")
                last_idx = idx

            qq = True
            at_depth_limit = False
            if last_idx == 0:
                f.write(self.prompt_pool["subquestion_prefix"].format(idx=self.n_shots+1, sub_idx=1))
            else:
                if last_idx % 2 == 0:
                    f.write(self.prompt_pool["subquestion_prefix"].format(idx=self.n_shots+1, sub_idx=math.ceil(last_idx/2)+1))
                    if (last_idx+1) >= self.depth_limit:
                        f.write(" " + self.prompt_pool["overall_question_prefix"])
                        at_depth_limit = True
                else:
                    f.write(self.prompt_pool["answer_prefix"].format(idx=self.n_shots+1, sub_idx=math.ceil(last_idx/2)))
                    qq = False
                
            model_input = f.getvalue()
        
        text_outputs = []
        tokens_prob = []
        gen_output = self.language_model.generate(prompt=model_input, num_return_sequences=n_actions)
        for output in gen_output.log_prob:
            text, logprob = self.extract_first_subquestion(output) if qq else self.extract_first_subanswer(output)
            text_outputs.append(text)
            tokens_prob.append(logprob)
        text_outputs = [output.strip() for output in text_outputs]

        if at_depth_limit:
            text_outputs = [self.prompt_pool["overall_question_prefix"] + ' ' + output for output in text_outputs]
        

        return text_outputs, tokens_prob
