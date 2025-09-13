from pydantic import BaseModel

class CacheArgs(BaseModel):
    cache_dir: str = 'cache'  # joblib缓存对话的目录（建议使用绝对路径）

cache_args = CacheArgs()