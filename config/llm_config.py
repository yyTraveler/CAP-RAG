from pydantic import BaseModel

class LLMArgs(BaseModel):
    api_key: str = "EMPTY"
    base_url: str = "http://127.0.0.1:13000/v1"
    max_retries: str = 5
    
llm_args = LLMArgs()