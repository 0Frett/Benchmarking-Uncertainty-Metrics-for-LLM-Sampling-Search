import os
import time
import threading
from queue import Queue
from typing import List
import openai
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from textwrap import dedent
import ipdb

load_dotenv()

OPENAI_API_MODELS = [
    'gpt-4o-mini', 
    'gpt-4o',
    'o3-mini'
]
VLM_MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
]

class GenerateOutput():
    """
    text: List[str]
        The generated text sequences from the model.
    log_prob: List[List[dict]]
        The log probabilities of the topk generated tokens of each text sequences.
        Each sub-list contains dictionaries with the main token and its top-k log probabilities.
        Each dictionary contains:
            - main_token: The main token. 
            - main_logprob: The log probability of the main token.
            - topk_logprobs: A list of log probabilities for the top-k tokens.
            - topk_tokens: A list of the top-k tokens.  
        For example:
            [
                [
                    {'main_token': 'Hello', 'main_logprob': -0.5, 'topk_logprobs': [-0.6, -0.7], 'topk_tokens': ['Hi', 'Hey']},
                    {'main_token': 'world', 'main_logprob': -0.4, 'topk_logprobs': [-0.5, -0.6], 'topk_tokens': ['earth', 'globe']}
                ],
                [
                    {'main_token': 'Goodbye', 'main_logprob': -0.3, 'topk_logprobs': [-0.4, -0.5], 'topk_tokens': ['Farewell', 'See you']},
                    {'main_token': 'everyone', 'main_logprob': -0.2, 'topk_logprobs': [-0.3, -0.4], 'topk_tokens': ['all', 'people']}
                ]
            ]
    """
    def __init__(self, text: List[str], log_prob: List[List[dict]]):
        self.text = text
        self.log_prob = log_prob


class OpenAIWorker(threading.Thread):
    def __init__(self, queue: Queue, model: str, api_key: str, max_tokens: int, rate_limit_per_min: int):
        threading.Thread.__init__(self)
        self.queue = queue
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limit_per_min = rate_limit_per_min

    def run(self):
        while True:
            prompt, num_return_sequences, temperature, retry, result_queue = self.queue.get()
            for i in range(1, retry + 1):
                try:
                    if self.rate_limit_per_min is not None:
                        time.sleep(60 / self.rate_limit_per_min)
 
                    if self.model in OPENAI_API_MODELS:
                        messages = [{"role": "user", "content": prompt}]
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=temperature,
                            n=num_return_sequences,
                            logprobs=True,
                            top_logprobs=20,
                        )

                        log_probs = [choice.logprobs.content for choice in response.choices]
                        content_match = []
                        for content in log_probs:
                            token_match = []
                            for tokenLogprob in content:
                                main_token = tokenLogprob.token
                                topk_logprobs = []
                                topk_tokens = []
                                for topk in tokenLogprob.top_logprobs:
                                    topk_logprobs.append(topk.logprob)
                                    topk_tokens.append(topk.token)

                                token_match.append(
                                    {
                                        'main_token': main_token, 
                                        'main_logprob': topk_logprobs[0],
                                        'topk_logprobs': topk_logprobs, 
                                        'topk_tokens': topk_tokens
                                    }
                                )
                            content_match.append(token_match)
                        texto = [choice.message.content for choice in response.choices]

                        result_queue.put(GenerateOutput(text=texto, log_prob=content_match))
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
        self.temperature = temperature

        for _ in range(num_workers):
            api_key = os.getenv("OPENAI_API_KEYS", "").split(',')[_%num_workers]
            worker = OpenAIWorker(self.queue, model, api_key, max_tokens, rate_limit_per_min=9999999)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        retry: int = 10
    ) -> GenerateOutput:
        
        result_queue = Queue()
        self.queue.put((prompt, num_return_sequences, self.temperature, retry, result_queue))
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    
    def batch_generate(
        self,
        prompts: List[str],
        num_return_sequences: int = 1,
        retry: int = 10
    ) -> List[GenerateOutput]:
        
        results = []
        result_queues = []
        for prompt in prompts:
            result_queue = Queue()
            result_queues.append(result_queue)
            self.queue.put((prompt, num_return_sequences, self.temperature, retry, result_queue))

        for result_queue in result_queues:
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        return results



class vlmModel():
    def __init__(self, model, max_tokens, temperature, tensor_parallel_size, gpu_memory_utilization, max_model_len=5000):
        self.model = LLM(
            model=model, 
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=64,
            task="generate",
            trust_remote_code=True, 
            enforce_eager=True
        )
        # Retrieve the tokenizer from the model
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = temperature
        # self.stop_token_ids = [self.tokenizer.eos_token_id]

    def generate(
        self, 
        prompt: str, 
        num_return_sequences: int = 1, 
    ) -> GenerateOutput:
        
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            logprobs=20,
            n=num_return_sequences,
            # repetition_penalty=1.1,
            # stop_token_ids=self.stop_token_ids
        )
        message = [{"role": "user", "content": prompt}]
        prompt_token_ids = self.tokenizer.apply_chat_template([message], add_generation_prompt=True)
        outputs = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)[0].outputs

        # outputs = self.model.generate(message, sampling_params)[0].outputs
        texto = []
        content_match = []
        for completion in outputs:
            texto.append(completion.text)
            token_match = []
            token_ids = completion.token_ids
            for idx, tokenLogprob in enumerate(completion.logprobs):
                main_token = tokenLogprob[token_ids[idx]].decoded_token
                main_logprob = tokenLogprob[token_ids[idx]].logprob
                # Sort logprobs by rank
                sorted_logprobs = sorted(tokenLogprob.values(), key=lambda x: x.rank)
                # Extract tokens and logprobs
                topk_logprobs = [logprob.logprob for logprob in sorted_logprobs]
                topk_tokens = [logprob.decoded_token for logprob in sorted_logprobs]
                token_match.append(
                    {
                        'main_token': main_token, 
                        'main_logprob': main_logprob,
                        'topk_logprobs': topk_logprobs,
                        'topk_tokens': topk_tokens,
                    }
                )
            content_match.append(token_match)

        return GenerateOutput(text=texto, log_prob=content_match)
    

    def batch_generate(
        self, 
        prompts: List[str], 
        num_return_sequences: int = 1, 
    ) -> List[GenerateOutput]:
        
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            logprobs=20,
            n=num_return_sequences,
        )
        messages = [[{"role": "user", "content": p}] for p in prompts]
        prompt_token_ids = [self.tokenizer.apply_chat_template(msg, add_generation_prompt=True) for msg in messages]
        res = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        gen_objs = [r.outputs for r in res]

        batch_outputs = []
        for outputs in gen_objs:
            texto = []
            content_match = []
            for completion in outputs:
                texto.append(completion.text)
                token_match = []
                token_ids = completion.token_ids
                for idx, tokenLogprob in enumerate(completion.logprobs):
                    main_token = tokenLogprob[token_ids[idx]].decoded_token
                    main_logprob = tokenLogprob[token_ids[idx]].logprob
                    # Sort logprobs by rank
                    sorted_logprobs = sorted(tokenLogprob.values(), key=lambda x: x.rank)
                    # Extract tokens and logprobs
                    topk_logprobs = [logprob.logprob for logprob in sorted_logprobs]
                    topk_tokens = [logprob.decoded_token for logprob in sorted_logprobs]
                    token_match.append(
                        {
                            'main_token': main_token, 
                            'main_logprob': main_logprob,
                            'topk_logprobs': topk_logprobs,
                            'topk_tokens': topk_tokens,
                        }
                    )
                content_match.append(token_match)
            batch_outputs.append(GenerateOutput(text=texto, log_prob=content_match))
        return batch_outputs


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    q = """
        Jerry is trying to cut down on the amount of soda he drinks. 
        Right now, he drinks 48 sodas a week. 
        If he cuts the number of sodas he drinks in half each week, 
        how many weeks will it take him to only drink 6 sodas per week?
    """
    prompts = [
        q,q,q
    ]
    # model = OpenAIModel_parallel(model='gpt-4.1-mini', temperature=0.7, max_tokens=128)
    # gen = model.batch_generate(prompts=prompts)
    

    model = vlmModel(
        # /model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 
        # model='mistralai/Mistral-7B-Instruct-v0.3',
        model='meta-llama/Meta-Llama-3-8B-Instruct',
        max_tokens=512, 
        temperature=0.8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )
    
    gen = model.batch_generate(prompts=prompts)
    ipdb.set_trace()
