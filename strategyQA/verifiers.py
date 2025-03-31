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
    def __init__(self, temperature:float=0.8, max_tokens: int = 200):
        self.model = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
            gpu_memory_utilization=0.5,
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


class llLlamaModel():
    def __init__(self, model_path, temperature: float = 1.0, max_tokens: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

        # Ensure that a pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, num_return_sequences: int = 1):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Tokenize using the chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Handle both dictionary and tensor return types
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)
        else:
            input_ids = inputs.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        # Define terminators for generation (if required)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  # pass attention mask
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id  # set pad_token_id explicitly
        )

        responses = []
        for output in outputs:
            # Slice off the prompt tokens; note: input_ids.shape[-1] gives the prompt length
            response = output[input_ids.shape[-1]:]
            text_res = self.tokenizer.decode(response, skip_special_tokens=True)
            responses.append(text_res)

        return responses

    def batch_generate(self, prompts: list):
        batch_messages = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            batch_messages.append(messages)
        
        # Tokenize inputs using chat template
        inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,  # Enable padding for batch processing
        )
        
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)
        else:
            input_ids = inputs.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        batch_responses = []
        for i, output in enumerate(outputs):
            prompt_length = input_ids.shape[-1]
            response = output[prompt_length:]
            text_res = self.tokenizer.decode(response, skip_special_tokens=True)
            batch_responses.append(text_res)
        
        return batch_responses
