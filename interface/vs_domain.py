"""schema for vector store
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import FieldSchema, MilvusClient, DataType, CollectionSchema, Function, FunctionType
from tqdm import tqdm
from .entity import Sample, SubNote, Note

logger = logging.getLogger(__name__)


class CAPSchema(BaseModel):
    pk: str = ''
    text: str = ''
    doc_id: str = ''
    label: str = ''  # query, summary, reference
    label_num: int = 0  # 用于区分是整体（ 0 ）还是细节（ >0 ）
    chunk_id: int = 0  # chunk 排序序号
    
    chunk_ids: str = '' # 聚类分块的聚类对象chunk_id合集
    chunk_ids_list: List[int] = Field(default_factory=list) # 聚类分块的聚类对象chunk_id合集
    detail_sign: int = 0 # 专用的标记位，和label_num的逻辑保持一致 0:整体，1:细节

class MilvusDao:
    
    def __init__(self, col_name: str) -> None:
        self.client = MilvusClient(uri="http://localhost:19530")
        self.col_name = col_name
    
    def list_all(self) -> List[Sample]:
        """数据全部加载到内存

        Returns:
            List[Sample]: _description_
        """
        samples: List[Sample] = []
        chunks = self.client.query(collection_name=self.col_name, 
                          filter="label == 'reference'",
                          output_fields=["pk", "text", "label", "chunk_seq", "doc_id", "chunk_id", "chunk_ids", "detail_sign"])
        for each in tqdm(chunks, total=len(chunks), desc="loading chunks: "):
            doc_id = each['doc_id']
            chunk_seq = each['chunk_seq']
            chunk = each['text']
            cluster_sign = each['chunk_ids'] != ''
            infos = self.client.query(collection_name=self.col_name,
                                      filter=f"doc_id == '{doc_id}' and chunk_seq == {chunk_seq}",
                                      output_fields=["pk", "text", "label", "chunk_seq", "doc_id", "chunk_id", "chunk_ids", "detail_sign"])
            label_2_item: dict[str, str] = {}
            s = Sample(chunk=chunk, cluster_sign=cluster_sign)
            s.doc_id = doc_id
            s.chunk_id = chunk_seq
            s.chunk_ids = each['chunk_ids']
            note = s.note
            max_index = 0
            for i in infos:
                t_label: str = i['label'].replace('ponit', 'point')
                label_2_item[t_label] = i['text']
                if 'point' in t_label:
                    max_index = max(max_index, int(t_label.split('_')[-1]))
            note.query = label_2_item['query']
            note.reference = label_2_item['content']
            note.summary = label_2_item['summary']
            for i in range(max_index + 1):
                query = label_2_item.get(f"point_query_{i}", '')
                summary = label_2_item.get(f"point_summary_{i}", '')
                reference = label_2_item.get(f"point_content_{i}", '')
                if query and summary and reference:
                    sub = SubNote(
                        query=query,
                        summary=summary,
                        reference=reference
                    )
                    note.points.append(sub)
            samples.append(s)
        return samples
        

    def create_collection(self, dimension: int = 768, desc: str = "", metric_type: str = 'COSINE'):
        """创建集合
        """
        
        if not self.client.has_collection(self.col_name):
            schema = CollectionSchema(
                auto_id = False,
                enable_dynamic_field = False,
                fields=[
                    FieldSchema(name='pk', dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False), # f"{doc_id}_{chunk_seq}_{label}_{label_num}"
                    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name='label', dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name='label_num', dtype=DataType.INT64),
                    FieldSchema(name='doc_id', dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name='chunk_id', dtype=DataType.INT64),
                    FieldSchema(name='chunk_ids', dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name='detail_sign', dtype=DataType.INT8),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
                ],
                description=desc,
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", 
                                   index_type="HNSW", 
                                   metric_type=metric_type, 
                                   M=8,
                                   efConstruction=64
                                   )
            self.client.create_collection(
                collection_name=self.col_name,
                dimension=dimension,
                primary_field_name='pk',
                # id_type=DataType.VARCHAR,
                metric_type='COSINE',
                schema=schema,
                index_params=index_params,
            )
        else:
            raise RuntimeError(f"collection {self.col_name} already exists.")
            
        
    def search(self, embedding: List[float], k: int, filter: str, metric_type: str = "COSINE") -> List[CAPSchema]:
        """查询
        """
        result = self.client.search(
            collection_name=self.col_name,
            data=[embedding],
            limit=k,
            filter=filter,
            search_params={"metric_type": metric_type, "params": {"ef": 1000 }},
            output_fields=["pk", "text", "label", "label_num", "doc_id", "chunk_id", "chunk_ids", "detail_sign"]
        )
        ans = []
        res_list = result[0]
        res_list.sort(key=lambda x: x.distance, reverse=False)
        for res in res_list:
            each = res['entity']
            ans.append(CAPSchema(
                pk=each['pk'],
                text=each['text'],
                doc_id=each['doc_id'],
                label=each['label'],
                label_num=each['label_num'],
                chunk_id=each['chunk_id'],
                chunk_ids=each['chunk_ids'],
                detail_sign=each['detail_sign'],
            ))
        return ans

    def search_by_pks(self, pks: List[str]) -> List[CAPSchema]:
        """根据主键查询
        """
        pks_str = "','".join(pks)
        result = self.client.query(
            collection_name=self.col_name,
            filter=f"pk in ['{pks_str}']",
            output_fields=["pk", "text", "label", "label_num", "doc_id", "chunk_id", "chunk_ids", "detail_sign"],
        )
        ans: List[CAPSchema] = []
        for each in result:
            ans.append(CAPSchema(
                pk=each['pk'],
                text=each['text'],
                doc_id=each['doc_id'],
                label=each['label'],
                label_num=each['label_num'],
                chunk_id=each['chunk_id'],
                chunk_ids=each['chunk_ids'],
                detail_sign=each['detail_sign'],
            ))
        ans.sort(key=lambda x: x.chunk_id)
        return ans
    
    def insert(self, schemas: List[CAPSchema], embedding: List[List[float]]):
        """插入数据
        """
        if not self.client.has_collection(self.col_name):
            raise ValueError(f"collection {self.col_name} not exists.")
        
        datas = []
        for index in tqdm(range(len(schemas)), total=len(schemas), desc="inserting schemas: ", disable=False):
            data = schemas[index]
            record = {
                'pk': f"{data.doc_id}_{data.chunk_id}_{data.label}_{data.label_num}_{data.detail_sign}",
                'text': data.text,
                'label': data.label,
                'label_num': data.label_num,
                'doc_id': data.doc_id,
                'chunk_id': data.chunk_id,
                'chunk_ids': data.chunk_ids,
                'detail_sign': data.detail_sign,
                "vector": embedding[index]
            }
            datas.append(record)
            if len(datas) == 500:
                self.client.insert(
                    collection_name=self.col_name,
                    data=datas
                )
                datas = []
        if datas:
            self.client.insert(
                collection_name=self.col_name,
                data=datas
            )
        


    
    
