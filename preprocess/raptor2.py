"""raptor 层次summary的实现，也就是官方的策略"""
from typing import List

from vllm import LLM, SamplingParams
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .raptor import group_chunks
from interface.chat import LLMChat
from interface.models import get_embedding_model
from config.pre_config import raptor_args


def batch_summary(chunks: List[Document]) -> List[str]:
    """生成文档摘要"""
    return LLMChat.batch_summary(raptor_args.model_name, [c.page_content for c in chunks])
    
def extract_page_content(docs: List[Document]) -> List[str]:
    """从文档中提取文本内容"""
    return [doc.page_content for doc in docs]

def recursive_abstractive_processing(
    embedding_model_name: str,
    chunks: List[Document],
) -> List[str]:
    """递归抽象处理，返回处理后的文档列表"""
    embedding_model = get_embedding_model(embedding_model_name)
    ans = [each.page_content for each in chunks]
    grouped_chunks = group_chunks(embedding_model, chunks, False)
    sums = batch_summary(grouped_chunks)
    ans.extend(sums)
    
    chunks = sums
    while chunks:
        grouped_chunks = group_chunks(embedding_model, chunks, False)
        if len(grouped_chunks) == 0 or len(grouped_chunks) >= len(chunks):
            break
        sums = batch_summary(grouped_chunks)
        ans.extend(sums)
        chunks = sums
    return ans