import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional
import torch
import tiktoken
from vllm import LLM, SamplingParams
from llm_utils import OPENAI_API_MODELS, VLM_MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


import ipdb
load_dotenv()

OPENAI_API_ENCODER = [
    'text-embedding-3-large',
    'text-embedding-3-small'
]
HF_ENCODER = [
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
VLM_ENCODER = [
    'intfloat/multilingual-e5-large',
    'jinaai/jina-embeddings-v3'
]
LLAMA_SUPPORT_QUANTIZED_MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct'
]
GEMMA_SUPPORT_QUANTIZED_MODELS = [
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it'
]

class openaiEmbedModel():
    def __init__(
        self, 
        model: str
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY1", ""))
        self.model_name = model
    
    def process_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        
        embed_result = self.client.embeddings.create(input=texts, model=self.model_name)
        output = [list(vec.embedding) for vec in embed_result.data]
        return output


class HiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        max_length: Optional[int] = 1024
    ):

        # model initialization
        if model_name in LLAMA_SUPPORT_QUANTIZED_MODELS:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type= "nf4")
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=quantization_config)
            max_batch_size = 16
            
        elif model_name in GEMMA_SUPPORT_QUANTIZED_MODELS:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", 
                quantization_config=quantization_config)
            max_batch_size = 16
            
        else:
            quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            max_batch_size = 4

        # Tokenizer initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = quantized_model
        self.max_length = max_length
        self.max_batch_size = max_batch_size

    def mean_pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies mean pooling to the last hidden state, taking the attention mask into account.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def process_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Tokenizes input texts, moves inputs to the correct device for embeddings,
        and computes mean-pooled embeddings.
        """
        # 1) Tokenize on CPU
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # 2) Move inputs to the embedding layer's device
        embed_device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}

        # 3) Forward pass (accelerate will handle sharding/offload)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # 4) Mean-pool last hidden state and return on CPU
        pooled = self.mean_pooling(
            outputs.hidden_states[-1],
            inputs["attention_mask"].to(outputs.hidden_states[-1].device)
        )
        return pooled.detach().cpu().tolist()



class vlmEmbedModel():
    def __init__(self, model, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        self.model = LLM(
            model=model, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            task="embed",
            trust_remote_code=True, 
            enforce_eager=True
        )
        self.max_batch_size = 32

    def process_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        
        texts = [f"{text}" for text in texts]
        outputs = self.model.embed(texts)
        vecs = [list(output.outputs.embedding) for output in outputs]
        return vecs

        


class model_tokenizer:
    def __init__(
        self,
        model_name: str,
    ):
        if model_name in OPENAI_API_MODELS:
            self.m = tiktoken.encoding_for_model(model_name)
        elif model_name in VLM_MODELS:
            self.m = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError('Unsupported Model Name')
    
    def tokenize(
        self, 
        text: str
    ) -> List[str]:
        
        ids = self.m.encode(text.lower())
        return [str(id) for id in ids]
        

if __name__ == "__main__":
    # extractor = HiddenStateExtractor(
    #     model_name="google/gemma-2-2b-it",
    #     max_length=1024
    # )

    # batch = ["""
    #     Jerry is trying to cut down on the amount of soda he drinks. 
    #     Right now, he drinks 48 sodas a week. 
    #     If he cuts the number of sodas he drinks in half each week, 
    #     how many weeks will it take him to only drink 6 sodas per week?
    # """]*3 + ["how are you"]
    # # returns a tensor of shape (3, seq_len, hidden_size)
    # last_hs1 = extractor.process_batch(batch)

    # mm = openaiEmbedModel(model='text-embedding-3-large')
    # last_hs2 = mm.process_batch(batch)
    from metric_utils import lexical_distance
    p = "Jerry is trying to cut down on the amount of soda he drinks."
    q = "Jerry is trying to how many weeks will it take him to only drink 6 sodas per week?"
    # aa = model_tokenizer('text-embedding-3-large')
    bb = model_tokenizer('google/gemma-2-2b-it')
    # atok = aa.tokenize(p)
    btok1 = bb.tokenize(p)
    btok2 = bb.tokenize(q)
    a = lexical_distance(btok1, btok2, "rouge-l")
    b = lexical_distance(btok1, btok2, "bleu-n")


    ipdb.set_trace()
