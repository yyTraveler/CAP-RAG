import os
import logging
from typing import List, Tuple
from tqdm import tqdm
import time
import json
from math import ceil

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from ..config.pre_config import raptor_args
from .raptor2 import recursive_abstractive_processing
from interface.entity import Sample, Note, SubNote
from interface.vs_domain import CAPSchema, MilvusDao
from interface.models import get_tokenizer, get_embedding_model, get_llm
from dataset.config import DatasetConfig, DatasetNameEnum
from dataset.quality.handler import QualityDatasetParser

logger = logging.getLogger(__name__)


class RAPTORPreProcessor:
    """
    1. 加载数据集 并 统一为CAPSchema
    2. embedding表示
    3. 入库
    """
    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.dataset_config = dataset_config
        
        # 加载模型
        self.tokenizer = get_tokenizer(raptor_args.embedding_model_name)
        self.text_splitter_main = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=raptor_args.chunk_size,
            chunk_overlap=raptor_args.chunk_overlap
        )
        
        # 维护数据库链接
        self.vs = MilvusDao(col_name=raptor_args.vs_col_name)
        
    def _process_quality(self) -> List[Sample]:
        """加载quality数据集的数据，但是保留note部分为默认值

        Returns:
            List[Sample]: _description_
        """
        dataset = QualityDatasetParser(profile=self.dataset_config.profile.value)
        datas = dataset.load()
        ans: List[Sample] = []
        for item in tqdm(datas, total=len(datas), desc="clustering quality dataset"):
            doc_id = item.article_id
            doc_content = item.cleaned_doc
            item_document = Document(id=doc_id, page_content=doc_content)
            origin_chunks = self.text_splitter_main.split_documents([item_document])
            chunks = recursive_abstractive_processing(
                embedding_model_name=raptor_args.embedding_model_name,
                chunks=origin_chunks
            )
            for index, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="clustering each doc: ", leave=True, disable=True):
                sample = Sample()
                sample.doc_id = item.article_id
                sample.chunk = chunk
                ans.append(sample)
        return ans 
            
    def _trans_cap_schema(self, samples: List[Sample]) -> List[CAPSchema]:
        """将数据转换为CAPSchema格式"""
        ans: List[CAPSchema] = []
        for index, each in enumerate(samples):
            ans.append(CAPSchema(
                doc_id=each.doc_id,
                label="",
                chunk_id=index,
                text=each.chunk
            ))
        return ans
        
    def _save_to_local(self, samples: List[Sample]) -> str:
        """将 samples 序列化以json的形式保存的本地

        Args:
            samples (List[CAPSchema]): _description_

        Returns:
            str: 保存数据的本地目录
        """
        base_dir = raptor_args.output_path
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dir_name = os.path.join(base_dir, current_time)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        with open(os.path.join(dir_name, "samples.json"), "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + '\n')
        
        with open(os.path.join(dir_name, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(raptor_args), f)
        
        logger.info(f"{len(samples)} sample data written to {dir_name}")
        return dir_name
            
    def run(self):
        logger.info(f"Running preprocessor for dataset: {self.dataset_config.name}")
        # 加载数据集并聚类
        if self.dataset_config.name == DatasetNameEnum.quality.value:
            datas = self._process_quality()
        else:
            raise Exception("please check your dataset enum.")
        schemas = self._trans_cap_schema(samples=datas)
        
        # 入库
        self.vs.create_collection(dimension=768, desc=raptor_args.vs_col_description)
        embeddings = []
        batch_size = 100
        if batch_size > len(schemas): batch_size = len(schemas)
        for i in range(0, len(schemas), batch_size):
            batch = schemas[i:i + batch_size if i + batch_size < len(schemas) else len(schemas)]
            texts = [each.text for each in batch]
            batch_embeddings = self.embedding_model.embed_documents(texts)
            embeddings.extend(batch_embeddings)
        self.vs.insert(schemas=schemas, embedding=embeddings)
        
        logger.info(f"Preprocessor finished for dataset: {self.dataset_config.name}")
        