from pydantic import BaseModel, Field

class RetrievalArgs(BaseModel):
    sign: bool = True  # 是否启用
    
    embedding_model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'  # 编码模型
    vs_col_name: str = ''  # milvus数据集的名称
    metric_type: str = 'L2'
    
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'  # 生成使用的模型
    concurrency: int = 2  # 单次处理的最大并发数量
    
    output_path: str = ''  # 输出目录的绝对路径， 没有指定就不输出(目录而不是文件名称)
    top_k: int = 5  # 检索的top_k数量
    type: int = 0  # 见Retrieval 0:传统检索，1:回溯检索，
    
retrieval_args = RetrievalArgs()