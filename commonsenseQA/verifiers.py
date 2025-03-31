import os
import time
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import torch
from vllm import LLM, SamplingParams

load_dotenv()

class OpenAIModel():
    def __init__(self, model:str, max_tokens:int = 256, temperature: float = 1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY2"))
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.temperature = temperature
    
    def generate(self, prompt: str, num_return_sequences: int = 1, rate_limit_per_min: int = 20, retry: int = 10):
        for i in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                if ('gpt-4o-mini' in self.model) or ('gpt-3.5-turbo' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        n=num_return_sequences,
                    )
                    self.completion_tokens += response.usage.completion_tokens
                    self.prompt_tokens += response.usage.prompt_tokens

                    text_responses = [choice.message.content for choice in response.choices]

                    return text_responses
                else:
                    print(f"Wrong Model Name !!!")
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries")


class LlamaModel():
    def __init__(self, temperature:float=0.5, max_tokens: int = 200):
        self.model = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
            gpu_memory_utilization=0.9,
            max_model_len=1024
        )
        self.temperature=temperature 
        self.max_tokens=max_tokens

    
    def generate(self, prompt: str, num_return_sequences: int = 1):
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            n=num_return_sequences
        )
        outputs = self.model.generate([prompt], sampling_params)[0].outputs
        texto = []
        for completion in outputs:
            texto.append(completion.text)

        return texto


class GemmaModel():
    def __init__(self, temperature:float=0.15, max_tokens: int = 200):
        self.model = LLM(
            model="google/gemma-2-9b-it", 
            gpu_memory_utilization=0.9,
            max_model_len=1024
        )
        self.temperature=temperature 
        self.max_tokens=max_tokens
    
    def generate(self, prompt: str, num_return_sequences: int = 1):
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            n=num_return_sequences
        )
        outputs = self.model.generate([prompt], sampling_params)[0].outputs
        texto = []
        for completion in outputs:
            texto.append(completion.text)
            
        return texto
