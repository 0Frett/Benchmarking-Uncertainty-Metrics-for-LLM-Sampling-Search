import os
import openai
import numpy as np
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import time
from openai import OpenAI

class GenerateOutput():
    def __init__(self, text: List[str], log_prob: List[List],):
        self.text = text
        self.log_prob = log_prob

class OpenAIModel():
    def __init__(self, model:str, max_tokens:int = 500):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY", None),
        )
        self.completion_tokens = 0
        self.prompt_tokens = 0
    
    def generate(self,
                prompt: str,
                num_return_sequences: int = 1,
                rate_limit_per_min: int = 20,
                temperature: float = 1.0,
                retry: int = 10,) -> GenerateOutput:
        
        for i in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                if ('gpt-3.5-turbo' in self.model) or ('gpt-4' in self.model) or ('gpt-4o' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        n=num_return_sequences,
                        logprobs=True,
                        top_logprobs=2,
                    )
                    self.completion_tokens += response.usage.completion_tokens
                    self.prompt_tokens += response.usage.prompt_tokens

                    log_probs = [choice.logprobs.content for choice in response.choices]
                    content_match = []
                    for content in log_probs:
                        token_match = []
                        for tokenLogprob in content:
                            token = tokenLogprob.token
                            top1Logprob = tokenLogprob.top_logprobs[0].logprob
                            top2Logprob = tokenLogprob.top_logprobs[1].logprob
                            token_match.append({'token':token, 
                                                'top1Logprob':top1Logprob, 
                                                'top2Logprob':top2Logprob})
                        content_match.append(token_match)
                    texto = [choice.message.content for choice in response.choices]

                    return GenerateOutput(
                        text=texto,
                        log_prob=content_match
                    )
                else:
                    print(f"Wrong Model Name !!!")
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries")

    def usage(self):
        if self.model == "gpt-4":
            cost = self.completion_tokens / 1000 * 0.06 + self.prompt_tokens / 1000 * 0.03
        elif self.model == "gpt-3.5-turbo":
            cost = self.completion_tokens / 1000 * 0.002 + self.prompt_tokens / 1000 * 0.0015
        elif self.model == "gpt-4o":
            cost = self.completion_tokens / 1000 * 0.03 + self.prompt_tokens / 1000 * 0.015
        print(f"model: {self.model}, completion_tokens: {self.completion_tokens}, prompt_tokens: {self.prompt_tokens}, cost: {cost}")

if __name__ == '__main__':
    model = OpenAIModel(model='gpt-3.5-turbo')
    print(model.generate(prompt='Hello, how are you?', temperature=2.0).text)