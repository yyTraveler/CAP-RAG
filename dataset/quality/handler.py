import os
import json
import logging
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_community.document_loaders import BSHTMLLoader

logger = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(__file__)

DATA_PATH = os.path.join(DIR_PATH, 'dataset')
TEMP_PATH = os.path.join(DIR_PATH, 'temp')
os.makedirs(name=TEMP_PATH, exist_ok=True)


class QualityQuestionItem(BaseModel):
    question: str
    question_unique_id: str
    options: List[str]
    difficult: int
    writer_label: int = -1# 测试集没有该数据
    gold_label: int = -1# 测试集没有该数据
    predict_label: int = -1 # 预测值，纯自定义字段
    total_related_chunks: int = -1
    total_chunks: int = -1


class QualityItem(BaseModel):
    article_id: str
    set_unique_id: str
    batch_num: str
    writer_id: str
    source: str
    title: str
    year: int | None
    author: str
    topic: str
    article: str
    questions: List[QualityQuestionItem]
    cleaned_doc: Optional[str] = ''
    
class QualityPredictItem(BaseModel):
    doc_id: str = ''
    question_id: str = ''
    question: str = ''
    options: List[str] = Field(default_factory=list)
    predict_label: int = -1
    gold_label: int = -1
    predicted_evidence: List[str] = Field(default_factory=list)
    

class QualityDatasetParser():
    """DataParser for Quality
    Quality数据集中，对于任意一篇文章(artical_id) 都有两个打标签的工作人员进行打标工作。
    所以每个artical_id都可能存在两条数据，这两条数据除了questions和set_unique_id，其他内容都是相同的
    """
    
    def __init__(self, base_path: str = DATA_PATH, profile: str = 'train') -> None:
        self.dataset_name = f'QuALITY.v1.0.1.{profile}'
        self.dataset_path = os.path.join(base_path, self.dataset_name)

    def load(self) -> List[QualityItem]:
        """整合当前数据集内的所有数据，合并artical_id相同的数据.
        使用 BSHTMLLoader 解析器对样本数据做简单清洗

        Returns:
            List[QualityItem]: _description_
        """
        ans = []
        logger.debug(f"dataset_path: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            datas = f.readlines()
        
        logger.debug(f"dataset size: {len(datas)}")
        article_id_2_item = {}  # type: Dict[str, QualityItem]
        for i in datas:
            i = i.strip()
            json_dict = json.loads(i)
            item = QualityItem.model_validate(json_dict)
            
            article_id = item.article_id  # quality 的 article_id 是一定会重复出现的
            if article_id_2_item.__contains__(article_id):
                existed_item = article_id_2_item.get(article_id)
                if existed_item:
                    existed_item.questions.extend(item.questions)
            else:
                html_path = os.path.join(TEMP_PATH, article_id)
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(item.article)
                
                html_loader = BSHTMLLoader(file_path=html_path)
                full_docs = html_loader.load()
                item.cleaned_doc = full_docs[0].page_content
                article_id_2_item.__setitem__(article_id, item)
                ans.append(item)
        return ans


def _test_main():
    _parser = QualityDatasetParser(profile='dev')
    datas = _parser.load()
    print(f"length: {len(datas)}")
    
if __name__ == '__main__':
    _test_main()
    