import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface.vs_domain import CAPSchema, MilvusDao
from interface.models import get_embedding_model

DST_COL_NAME = "quality_preparse_official"
DATASOURCE_PATH = "cap-rag-official.json"


def main():
    with open(DATASOURCE_PATH, "r", encoding="utf-8") as f:
        datas = json.load(f)
    
    cap_schemas = [CAPSchema.model_validate(d) for d in datas]
    print(f"Loaded {len(cap_schemas)} records from JSON.")
    page_size = 1000

    # 获取总数
    total = len(cap_schemas)
    dst_dao = MilvusDao(DST_COL_NAME)
    try:
        dst_dao.create_collection(metric_type='L2')
    except Exception as e:
        print(f"Collection may already exist: {e}")

    embedding_model = get_embedding_model(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cuda") 
    for i in tqdm(range(0, total, page_size), desc="Importing"):
        batch = cap_schemas[i : i + page_size]
        embeddings = []
        texts = [each.text for each in batch]
        batch_embeddings = embedding_model.embed_documents(texts)
        embeddings.extend(batch_embeddings)
        dst_dao.insert(batch, embeddings)

if __name__ == "__main__":
    main()