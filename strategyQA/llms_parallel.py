import os
import time
import threading
from queue import Queue
from typing import List
import openai
from dotenv import load_dotenv
# from vllm import LLM, SamplingParams
load_dotenv()
 
class GenerateOutput():
    def __init__(self, text: List[str], log_prob: List[List],):
        self.text = text
        self.log_prob = log_prob

class OpenAIWorker(threading.Thread):
    def __init__(self, queue: Queue, model: str, api_key: str, max_tokens: int, rate_limit_per_min: int):
        threading.Thread.__init__(self)
        self.queue = queue
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key=api_key,
        )
        self.rate_limit_per_min = rate_limit_per_min

    def run(self):
        while True:
            prompt, num_return_sequences, temperature, retry, result_queue = self.queue.get()
            for i in range(1, retry + 1):
                try:
                    if self.rate_limit_per_min is not None:
                        time.sleep(60 / self.rate_limit_per_min)
 
                    if ('gpt-3.5-turbo' in self.model) or ('gpt-4o-mini' in self.model):
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

                        log_probs = [choice.logprobs.content for choice in response.choices]
                        content_match = []
                        for content in log_probs:
                            token_match = []
                            for tokenLogprob in content:
                                token = tokenLogprob.token
                                top1Logprob = tokenLogprob.top_logprobs[0].logprob
                                top2Logprob = tokenLogprob.top_logprobs[1].logprob
                                token_match.append({'token': token, 
                                                    'top1Logprob': top1Logprob, 
                                                    'top2Logprob': top2Logprob})
                            content_match.append(token_match)
                        texto = [choice.message.content for choice in response.choices]

                        result_queue.put(GenerateOutput(
                            text=texto,
                            log_prob=content_match
                        ))
                        self.queue.task_done()
                        break
                    else:
                        print(f"Wrong Model Name !!!")
                except Exception as e:
                    print(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            else:
                result_queue.put(RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries"))
                self.queue.task_done()

class OpenAIModel_parallel():
    def __init__(self, model: str, temperature:float, max_tokens: int = 500, num_workers: int = 2):
        self.model = model
        self.max_tokens = max_tokens
        self.queue = Queue()
        self.num_workers = num_workers
        self.workers = []
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.temperature = temperature

        for _ in range(num_workers):
            api_key = os.getenv("OPENAI_API_KEYSS", "").split(',')[_%num_workers]
            worker = OpenAIWorker(self.queue, model, api_key, max_tokens, rate_limit_per_min=9999999)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def generate(self,
                 prompt: str,
                 num_return_sequences: int = 1,
                 rate_limit_per_min: int = 8000,
                 retry: int = 10) -> GenerateOutput:
        
        result_queue = Queue()
        self.queue.put((prompt, num_return_sequences, self.temperature, retry, result_queue))
        result = result_queue.get()

        if isinstance(result, Exception):
            raise result

        self.completion_tokens += sum(len(output) for output in result.text)
        self.prompt_tokens += len(prompt.split())

        return result

    def usage(self):
        if self.model == "gpt-4":
            cost = self.completion_tokens / 1000 * 0.06 + self.prompt_tokens / 1000 * 0.03
        elif self.model == "gpt-3.5-turbo":
            cost = self.completion_tokens / 1000 * 0.002 + self.prompt_tokens / 1000 * 0.0015
        elif self.model == "gpt-4o":
            cost = self.completion_tokens / 1000 * 0.03 + self.prompt_tokens / 1000 * 0.015
        print(f"model: {self.model}, completion_tokens: {self.completion_tokens}, prompt_tokens: {self.prompt_tokens}, cost: {cost}")


class LlamaModel():
    def __init__(self, temperature:float=0.8, max_tokens: int = 200):
        self.model = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
            gpu_memory_utilization=0.5
        )
        self.temperature=temperature 
        self.max_tokens=max_tokens

    
    def generate(self, prompt: str, num_return_sequences: int = 1):
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            logprobs=2,
            n=num_return_sequences
        )
        outputs = self.model.generate([prompt], sampling_params)[0].outputs
        texto = []
        content_match = []
        for completion in outputs:
            texto.append(completion.text)
            token_match = []
            for tokenLogprob in completion.logprobs:
                sorted_logprobs = sorted(tokenLogprob.values(), key=lambda x: x.rank)
                # Extract tokens and logprobs
                token = sorted_logprobs[0].decoded_token
                top1Logprob = sorted_logprobs[0].logprob
                top2Logprob = sorted_logprobs[1].logprob

                token_match.append(
                    {
                        'token': token, 
                        'top1Logprob': top1Logprob, 
                        'top2Logprob': top2Logprob
                    }
                )
            content_match.append(token_match)

        return GenerateOutput(text=texto, log_prob=content_match)


class GemmaModel():
    def __init__(self, temperature:float=0.15, max_tokens: int = 200):
        self.model = LLM(
            model="google/gemma-2-9b-it", 
            gpu_memory_utilization=0.5
        )
        self.temperature=temperature 
        self.max_tokens=max_tokens

    
    def generate(self, prompt: str, num_return_sequences: int = 1):
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            logprobs=2,
            n=num_return_sequences
        )
        outputs = self.model.generate([prompt], sampling_params)[0].outputs
        texto = []
        content_match = []
        for completion in outputs:
            texto.append(completion.text)
            token_match = []
            for tokenLogprob in completion.logprobs:
                sorted_logprobs = sorted(tokenLogprob.values(), key=lambda x: x.rank)
                # Extract tokens and logprobs
                token = sorted_logprobs[0].decoded_token
                top1Logprob = sorted_logprobs[0].logprob
                top2Logprob = sorted_logprobs[1].logprob

                token_match.append(
                    {
                        'token': token, 
                        'top1Logprob': top1Logprob, 
                        'top2Logprob': top2Logprob
                    }
                )
            content_match.append(token_match)

        return GenerateOutput(text=texto, log_prob=content_match)



class MistralModel():
    def __init__(self, temperature:float=0.15, max_tokens: int = 200):
        self.model = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            gpu_memory_utilization=0.5
        )
        self.temperature=temperature 
        self.max_tokens=max_tokens

    
    def generate(self, prompt: str, num_return_sequences: int = 1):
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            logprobs=2,
            n=num_return_sequences
        )
        outputs = self.model.generate([prompt], sampling_params)[0].outputs
        texto = []
        content_match = []
        for completion in outputs:
            texto.append(completion.text)
            token_match = []
            for tokenLogprob in completion.logprobs:
                sorted_logprobs = sorted(tokenLogprob.values(), key=lambda x: x.rank)
                # Extract tokens and logprobs
                token = sorted_logprobs[0].decoded_token
                top1Logprob = sorted_logprobs[0].logprob
                top2Logprob = sorted_logprobs[1].logprob

                token_match.append(
                    {
                        'token': token, 
                        'top1Logprob': top1Logprob, 
                        'top2Logprob': top2Logprob
                    }
                )
            content_match.append(token_match)

        return GenerateOutput(text=texto, log_prob=content_match)




if __name__ == '__main__':
    # model = OpenAIModel_parallel(model='gpt-3.5-turbo', temperature=0.7, max_tokens=100)
    # gen = model.generate(prompt='How to cook steak', num_return_sequences=6)
    # model = MistralModel()
    model = GemmaModel(temperature=0.1, max_tokens=100)
    gen = model.generate(prompt='Are more people today related to Genghis Khan than Julius Caesar?', num_return_sequences=2)
    print(gen.text)
    # print(gen.log_prob)
