from pydantic import BaseModel, Field

class PreArgs(BaseModel):
    # 消融参数
    cluster_sign: bool = True  # 是否启用聚类
    preparse_sign: bool = True  # 是否启用预解析
    
    # 预处理阶段的分块配置
    main_chunk_size: int =100  # 整体分块大小
    main_chunk_overlap: int = 10  # 整体分块重叠大小
    
    # 预解析的模型配置
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'  # 预解析使用的模型
    concurrency: int = 2  # 单次处理的最大并发数量
    output_path: str = ''  # 输出目录的绝对路径， 没有指定就不输出，指定这个目录可以以json形式保存一份预解析的结果
    
    # 向量化与持久化
    embedding_model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'  # 编码模型
    vs_col_name: str = ''  # milvus数据集的名称
    vs_col_description: str = ''  # milvus数据集的描述
    metric_type: str = 'COSINE'
    
class BM25Args(BaseModel):
    chunk_size: int = 100
    chunk_overlap: int = 10
    embedding_model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'  # 编码模型
    vs_col_name: str = ''  # milvus数据集的名称
    vs_col_description: str = ''  # milvus数据集的描述
    
class RAPTORArgs(BaseModel):
    chunk_size: int = 100
    chunk_overlap: int = 10
    embedding_model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'  # 编码模型
    vs_col_name: str = ''  # milvus数据集的名称
    vs_col_description: str = ''  # milvus数据集的描述
    
    max_clusters: int = 50  # 最大聚类数目
    random_state: int = 0  # 随机种子
    
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'  # 预解析使用的模型
    output_path: str = ''  # 输出目录的绝对路径， 没有指定就不输出，指定这个目录可以以json形式保存一份预解析的结果

pre_args = PreArgs()
raptor_args = RAPTORArgs()
