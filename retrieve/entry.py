import os
import json
from typing import List, Set
from tqdm import tqdm
import time
import logging

import torch
import asyncio
from dataset.config import DatasetConfig, DatasetNameEnum, ProfileEnum
from dataset.quality.handler import QualityDatasetParser, QualityItem, QualityQuestionItem, QualityPredictItem
from config.retrieve_config import retrieval_args
from interface.vs_domain import MilvusDao, CAPSchema
from interface.models import get_embedding_model, get_tokenizer
from interface.chat import LLMChat

logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.generator = None

        self.tokenizer = get_tokenizer(retrieval_args.embedding_model_name)
        self.embedding_model = get_embedding_model(retrieval_args.embedding_model_name)
        self.vs = MilvusDao(col_name=retrieval_args.vs_col_name)
        
    def _token_counter(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text)
        return len(token_ids)

    def retrieve_tradition(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        if 'dpr' in retrieval_args.embedding_model_name:
            input_ids = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)["inputs"]
            with torch.no_grad():
                e = self.embedding_model(input_ids).pooler_output
                query_embedding = e[0].cpu().numpy().tolist()
        else:
            query_embedding = self.embedding_model.embed_query(query)
        ans = self.vs.search(
            embedding=query_embedding, 
            k=k, 
            filter=f"doc_id == '{doc_id}' and label == 'reference' and label_num == 0 and chunk_ids == ''", 
            metric_type=metric_type)
        return ans

    def retrieve_900(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """专用于其他技术路线的检索器"""
        if 'dpr' in retrieval_args.embedding_model_name:
            input_ids = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)["input_ids"]
            with torch.no_grad():
                e = self.embedding_model(input_ids).pooler_output
                query_embedding = e[0].cpu().numpy().tolist()
        else:
            query_embedding = self.embedding_model.embed_query(query)
        ans = self.vs.search(
            embedding=query_embedding, 
            k=k, 
            filter=f"doc_id == '{doc_id}'", 
            metric_type=metric_type)
        return ans
    
    def retrieve_mock_callback(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """模拟小论文中的回溯实现，主要用于统计召回样本的类型"""
        query_embedding = self.embedding_model.embed_query(query)
        l1_candidates = self.vs.search(
            embedding=query_embedding, 
            k=k * 10, 
            filter=f"doc_id == '{doc_id}'", 
            metric_type=metric_type)
        # 回溯到原文
        ans_pks: Set[str] = set()
        source_list: List[CAPSchema] = []
        for candidate in l1_candidates:
            if len(ans_pks) >= k:
                break
            # 如果遇到重复的pk，正常是靠set的去重去掉的，这里保留下来因为这个分块起了作用
            if candidate.label == 'reference' and candidate.label_num == 0:
                ans_pks.add(candidate.pk)
                source_list.append(candidate)
            else:
                chunk_id = candidate.chunk_id
                ans_pks.add(f"{candidate.doc_id}_{chunk_id}_reference_0_0")
                source_list.append(candidate)
        return source_list
    
    def retrieve_callback(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """小论文中的回溯实现"""
        query_embedding = self.embedding_model.embed_query(query)
        l1_candidates = self.vs.search(
            embedding=query_embedding, 
            k=k * 20, 
            filter=f"doc_id == '{doc_id}'", 
            metric_type=metric_type)
        # 回溯到原文
        ans_pks: Set[str] = set()
        for candidate in l1_candidates:
            if len(ans_pks) >= k:
                break
            if candidate.chunk_ids:
                # 如果有chunk_ids，说明是分块的，需要回溯到原文
                chunk_ids = candidate.chunk_ids.split(',')
                for chunk_id in chunk_ids:
                    ans_pks.add(f"{candidate.doc_id}_{chunk_id}_reference_0_0")
            else:
                ans_pks.add(f"{candidate.doc_id}_{candidate.chunk_id}_reference_0_0")
        return self.vs.search_by_pks(list(ans_pks))
    
    def retrieve_crag(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """小论文中的消融实现，只检索原文分块和聚类分块"""
        query_embedding = self.embedding_model.embed_query(query)
        l1_candidates = self.vs.search(
            embedding=query_embedding, 
            k=k * 10, 
            filter=f"doc_id == '{doc_id}' and label_num == 0", 
            metric_type=metric_type)
        # 回溯到原文
        ans_pks: Set[str] = set()
        for candidate in l1_candidates:
            if len(ans_pks) >= k:
                break
            if candidate.label == 'reference' and candidate.label_num == 0:
                # 如果是reference类型的原文文本，直接添加
                ans_pks.add(candidate.pk)
            else:
                chunk_id = candidate.chunk_id
                ans_pks.add(f"{candidate.doc_id}_{chunk_id}_reference_0_0")
        return self.vs.search_by_pks(list(ans_pks))
    
    def retrieve_prag(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """小论文中的消融实现，只检索原文分块和预解析分块"""
        query_embedding = self.embedding_model.embed_query(query)
        l1_candidates = self.vs.search(
            embedding=query_embedding, 
            k=k * 10, 
            filter=f"doc_id == '{doc_id}' and chunk_ids == ''", 
            metric_type=metric_type)
        # 回溯到原文
        ans_pks: Set[str] = set()
        for candidate in l1_candidates:
            if len(ans_pks) >= k:
                break
            if candidate.label == 'reference' and candidate.label_num == 0:
                # 如果是reference类型的原文文本，直接添加
                ans_pks.add(candidate.pk)
            else:
                chunk_id = candidate.chunk_id
                ans_pks.add(f"{candidate.doc_id}_{chunk_id}_reference_0_0")
        return self.vs.search_by_pks(list(ans_pks))
    
    def retrieve_901(self, query: str, doc_id: str, k: int=5, metric_type: str='COSINE') -> List[CAPSchema]:
        """专用于raptor的检索器，这个是基于序列长度做的"""
        MAX_TOKENS = k * 100 # 注意这里默认其他规则的切分规则是100个token一个chunk
        query_embedding = self.embedding_model.embed_query(query)
        temp = self.vs.search(
            embedding=query_embedding, 
            k=k, 
            filter=f"doc_id == '{doc_id}'", 
            metric_type=metric_type)
        tokens_count = 0
        ans: List[CAPSchema] = []
        index = 0
        while index < len(temp) and tokens_count < MAX_TOKENS:
            ans.append(temp[index])
            tokens_count += self._token_counter(temp[index].text)
            index += 1
        return ans
    
    def route_retrieve(self, type: int, question: str, doc_id: str, k: int, metric_type: str) -> List[CAPSchema]:
        candidates = []
        if type == 0:
            # Retrieve using traditional method
            candidates = self.retrieve_tradition(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 1:
            # Retrieve using callback method
            candidates = self.retrieve_callback(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 2:
            candidates = self.retrieve_crag(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 3:
            candidates = self.retrieve_prag(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 101:
            candidates = self.retrieve_mock_callback(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 900:
            candidates = self.retrieve_900(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        elif type == 901:
            candidates = self.retrieve_901(query=question, doc_id=doc_id, k=k, metric_type=metric_type)
        else:
            pass
        return candidates

    def retrieve_quality(self, type: int = 0, k: int = 5, metric_type: str='COSINE'):
        dataset = QualityDatasetParser(profile=self.dataset_config.profile.value)
        datas = dataset.load()
        ans: List[QualityPredictItem] = []
        concurrency = retrieval_args.concurrency
        for data in tqdm(datas, total=len(datas), desc='Retrieving Quality: '):
            doc_id = data.article_id
            qas = data.questions
            for i in tqdm(range(0, len(qas), concurrency), total=(len(qas) + concurrency - 1) // concurrency, desc="retrieve samples in batch mode (concurrent): ", disable=False):
                batch_qas = qas[i:i + concurrency]
                candidate_contexts: list[list] = []
                queries: List[str] = []
                options: List[List[str]] = []
                for each_qas in batch_qas:
                    question_id = each_qas.question_unique_id
                    question = each_qas.question
                    candidates = self.route_retrieve(
                        type=type,
                        question=question,
                        doc_id=doc_id,
                        k=k,
                        metric_type=metric_type,
                    )
                    candidate_contexts.append([c.text for c in candidates])
                    queries.append(question)
                    options.append(each_qas.options)
                answers = asyncio.run(
                    LLMChat.batch_quality_chat(
                        model_name=retrieval_args.model_name,
                        contexts=candidate_contexts,
                        options=options,
                        queries=queries
                    ))
                for answer, qa, candi in zip(answers, batch_qas, candidate_contexts):
                    ans.append(QualityPredictItem(
                        doc_id=doc_id,
                        question_id=qa.question_unique_id,
                        question=qa.question,
                        options=qa.options,
                        predict_label=answer[0],
                        gold_label=qa.gold_label,
                        predicted_evidence=candi
                    ))
        if retrieval_args.output_path:
            base_dir = retrieval_args.output_path
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            dir_name = os.path.join(base_dir, current_time)
            logging.info(f"Saving results to {dir_name}")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(os.path.join(dir_name, 'quality_answer.json'), 'w', encoding="utf-8") as f:
                for each in ans:
                    f.write(each.model_dump_json() + '\n')
            with open(os.path.join(dir_name, 'config.json'), 'w', encoding="utf-8") as f:
                json.dump(vars(retrieval_args), f)
        logger.info(f"Quality retrieval and generation completed, total {len(ans)} items retrieved.")
        

