import os
import logging
from typing import List
from tqdm import tqdm
import time
import json
from math import ceil
import asyncio

import torch
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from config.pre_config import pre_args
from .raptor import group_chunks
from interface.prompts import preparse_prompt_schema
from interface.entity import Sample, Note, SubNote
from interface.vs_domain import CAPSchema, MilvusDao
from interface.chat import LLMChat
from interface.models import get_tokenizer, get_embedding_model
from dataset.config import DatasetConfig, DatasetNameEnum
from dataset.quality.handler import QualityDatasetParser, QualityItem, QualityQuestionItem

logger = logging.getLogger(__name__)


class PreProcessor:
    """
    1. 加载数据集 并 统一为CAPSchema
    2. embedding表示
    3. 入库
    """
    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.dataset_config = dataset_config
        
        # 加载模型
        self.tokenizer = get_tokenizer(pre_args.embedding_model_name)
        self.text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=pre_args.main_chunk_size,
            chunk_overlap=pre_args.main_chunk_overlap
        )
        
        # 维护数据库链接
        self.vs = MilvusDao(col_name=pre_args.vs_col_name)
        try:
            self.vs.create_collection(dimension=768, desc=pre_args.vs_col_description, metric_type=pre_args.metric_type)
        except Exception as e:
            pass
    
    def _prompt_both(self, chunk: str):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given context {chunk}, read and understand the context above and fill the form as the following json template: {preparse_prompt_schema}"}
        ]
    
    def _cluster_qulaity(self, datas: List[QualityItem]) -> List[Sample]:
        ans: List[Sample] = []
        for item in tqdm(datas, total=len(datas), desc="clustering quality dataset", disable=False):
            doc_id = item.article_id
            doc_content = item.cleaned_doc
            item_document = Document(id=doc_id, page_content=doc_content)
            chunks = self.text_splitter.split_documents([item_document])
            if pre_args.cluster_sign:
                chunks = group_chunks(embedding_model_name=pre_args.embedding_model_name, chunks=chunks)

            for index, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="clustering each doc: ", leave=True, disable=True):
                sample = Sample()
                sample.doc_id = item.article_id
                sample.chunk_id = index
                
                cluster_ids = chunk.metadata['chunk_ids'] if 'chunk_ids' in chunk.metadata.keys() else []
                sample.chunk_ids_list = cluster_ids
                
                chunk_ids = [str(_) for _ in cluster_ids]
                sample.chunk_ids = ",".join(chunk_ids)
                sample.cluster_sign = True if 'chunk_ids' in chunk.metadata.keys() else False
                sample.chunk = chunk.page_content
                ans.append(sample)
        return ans

    def _preparse_batch(self, samples: List[Sample]) -> List[CAPSchema]:
        """预解析， 生成query、summary和context三元组"""
        ans: List[CAPSchema] = []
        concurrency = pre_args.concurrency
        total = len(samples)
        
        for i in tqdm(range(0, total, concurrency), total=(total + concurrency - 1) // concurrency, desc="preparse samples in batch mode (concurrent): ", disable=False):
            batch = samples[i:i + concurrency]
            prompts = [self._prompt_both(s.chunk) for s in batch]
            
            notes = asyncio.run(LLMChat.batch_chunk_preparse(pre_args.model_name, prompts))
            for sample, note in zip(batch, notes):
                if note:
                    note.reference = sample.chunk
                    sample.note = note

        dir_path = ''
        if pre_args.output_path:
            dir_path = self._save_to_local(samples)

        ans = self._trans_cap_schema(samples)
        return ans, dir_path
            
    def _trans_cap_schema(self, samples: List[Sample]) -> List[CAPSchema]:
        """将数据转换为CAPSchema格式"""
        ans: List[CAPSchema] = []
        for sample in samples:
            for index, each in enumerate([sample.note] + sample.note.points):
                if each.query:
                    ans.append(CAPSchema(
                        doc_id=sample.doc_id,
                        label="query",
                        label_num=index,
                        chunk_id=sample.chunk_id,
                        text=each.query,
                        chunk_ids=sample.chunk_ids,
                        cluster_sign=sample.cluster_sign,
                        detail_sign=0 if index == 0 else 1
                    ))
                if each.summary:
                    ans.append(CAPSchema(
                        doc_id=sample.doc_id,
                        label="summary",
                        label_num=index,
                        chunk_id=sample.chunk_id,
                        text=each.summary,
                        chunk_ids=sample.chunk_ids,
                        cluster_sign=sample.cluster_sign,
                        detail_sign=0 if index == 0 else 1
                    ))
                if each.reference:
                    ans.append(CAPSchema(
                        doc_id=sample.doc_id,
                        label="reference",
                        label_num=index,
                        chunk_id=sample.chunk_id,
                        text=each.reference,
                        chunk_ids=sample.chunk_ids,
                        cluster_sign=sample.cluster_sign,
                        detail_sign=0 if index == 0 else 1
                    ))
        return ans
        
    def _save_to_local(self, samples: List[Sample]) -> str:
        """将 samples 序列化以json的形式保存的本地

        Args:
            samples (List[CAPSchema]): _description_

        Returns:
            str: 保存数据的本地目录
        """
        base_dir = pre_args.output_path
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dir_name = os.path.join(base_dir, current_time)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        with open(os.path.join(dir_name, "samples.json"), "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + '\n')
        
        with open(os.path.join(dir_name, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(pre_args), f)
        
        logger.info(f"{len(samples)} sample data written to {dir_name}")
        return dir_name
            
    def run(self):
        logger.info(f"Running preprocessor for dataset: {self.dataset_config.name}")
        # 1. 加载数据集并聚类
        samples: list[Sample] = []
        if self.dataset_config.name == DatasetNameEnum.quality.value:
            dataset = QualityDatasetParser(profile=self.dataset_config.profile.value)
            datas = dataset.load()
            samples = self._cluster_qulaity(datas)
        else:
            raise Exception("please check the dataset enum.")
        
        # 2. 预解析
        schemas: List[CAPSchema] = []
        dir_path= ''
        if pre_args.preparse_sign:
            schemas, dir_path = self._preparse_batch(samples)
            logging.info(''f"Preprocessing finished, data saved to {dir_path}")
        else:
            for each in samples:
                each.note.reference = each.chunk
            schemas.extend(self._trans_cap_schema(samples))
                
        # 3. 入库
        batch_size = 1000
        embedding_model = get_embedding_model(pre_args.embedding_model_name)
        if batch_size > len(schemas): batch_size = len(schemas)
        for i in tqdm(range(0, len(schemas), batch_size), total=ceil(len(schemas)/batch_size), desc="embedding: ", disable=False):
            batch = schemas[i:i + batch_size if i + batch_size < len(schemas) else len(schemas)]
            embeddings = []
            texts = [each.text for each in batch]
            if 'dpr' in pre_args.embedding_model_name:
                input_ids = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)["input_ids"]
                with torch.no_grad():
                    e = embedding_model(input_ids).pooler_output
                    embeddings.extend([ee.cpu().numpy().tolist() for ee in e])
            else:
                batch_embeddings = embedding_model.embed_documents(texts)
                embeddings.extend(batch_embeddings)
            # 每个batch直接入库，大批量入库有可能报错
            self.vs.insert(schemas=batch, embedding=embeddings)
        logger.info(f"Preprocessor finished for dataset: {self.dataset_config.name}")
        return dir_path
        