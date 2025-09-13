from functools import cache
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from vllm import LLM
from openai import AsyncOpenAI

from config.pre_config import pre_args
from config.llm_config import llm_args

@cache
def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    获取指定模型的tokenizer
    :param model_name: 模型名称
    :return: tokenizer
    """
    if "dpr" in model_name:
        if "ctx" in model_name:
            from transformers import DPRContextEncoderTokenizer
            return DPRContextEncoderTokenizer.from_pretrained(model_name,)
        elif "question" in model_name:
            from transformers import DPRQuestionEncoderTokenizer
            return DPRQuestionEncoderTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

@cache
def get_embedding_model(model_name: str, device: str = "cpu"):
    if "dpr" in model_name:
        if "ctx" in model_name:
            from transformers import DPRContextEncoder
            return DPRContextEncoder.from_pretrained(model_name, trust_remote_code=True)
        elif "question" in model_name:
            from transformers import DPRQuestionEncoder
            return DPRQuestionEncoder.from_pretrained(model_name, trust_remote_code=True)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True}
    )

@cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=llm_args.api_key,
        base_url=llm_args.base_url,
        max_retries=llm_args.max_retries
    )
    
# @cache
# def get_llm(model_name:str, tensor_parallel_size: int, max_num_seqs: int) -> LLM:
#     return LLM(
#             model=model_name, 
#             tensor_parallel_size=tensor_parallel_size,
#             max_num_seqs=max_num_seqs,
#             enable_lora=True,
#             seed=1024,
#             enable_sleep_mode=True,
#             max_lora_rank=32,
#             gpu_memory_utilization=0.8
            
#         )