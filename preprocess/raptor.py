from typing import List
import logging
from copy import deepcopy
from joblib import Memory

import numpy as np
from sklearn.mixture import GaussianMixture
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from interface.models import get_embedding_model
from config.cache_config import cache_args

memory = Memory(location=cache_args.cache_dir, verbose=0)

logging.getLogger().setLevel(logging.ERROR)

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """umap算法做降维操作，以加速后续聚类算法的运行
    """
    from umap import UMAP
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    if n_neighbors <= 1:
        return embeddings
    reduced_embeddings = UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 0
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    if bics:
        optimal_clusters = n_clusters[np.argmin(bics)]
    else:
        optimal_clusters = -1
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    if n_clusters <= 0:
        return [], -1
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

@memory.cache
def group_chunks(embedding_model_name: str, chunks: List[Document], keep_source: bool=True) -> List[Document]:
    """group the chunks and do tasks to generate new group nodes
    return chunks and grouped nodes in one list

    the chunks sent in should from a same article
    
    cite: https://github.com/parthsarthi03/raptor
    """
    embedding_model = get_embedding_model(embedding_model_name)
    chunk_embeddings = embedding_model.embed_documents(
        [i.page_content for i in chunks])

    # 降维、聚类
    np_chunk_embeddings = np.array([each for each in chunk_embeddings])
    reduced_embeddings_global = global_cluster_embeddings(
        np_chunk_embeddings, min(10, len(np_chunk_embeddings) - 2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, 0.1
    )
    if n_global_clusters <= 0:
        return []

    # 根据聚类的结果，组合chunk
    cluster_number_2_chunk_number = {}  # cluster—number  ==> chunk number
    chunk_number_2_cluser_number = []
    for index, each in enumerate(global_clusters):
        key = each[0]  # cluster_number
        chunk_number_2_cluser_number.append(str(key))
        compose_chunk_list = cluster_number_2_chunk_number.get(key, [])
        compose_chunk_list.append(index)
        cluster_number_2_chunk_number[key] = compose_chunk_list

    # 将聚类后的结果也封装到返回值里
    ans = []
    if keep_source:
        ans = deepcopy(chunks)
    for cluser_name, chunk_numbers in cluster_number_2_chunk_number.items():
        sub_str = "\n".join([chunks[i].page_content for i in chunk_numbers])
        ans.append(Document(page_content=sub_str, metadata={
            "chunk_ids": chunk_numbers
        }))
    return ans



        