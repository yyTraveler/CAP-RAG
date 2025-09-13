from pydantic import BaseModel, Field
from typing import List, Optional

class SubNote(BaseModel):
    query: str = Field(default='', description='propose a possiable question which is deep and could get the answer from the content field.')
    summary: str = Field(default='', description='the main idea of this point or the standard answer to the query.')
    reference: str = Field(default='', description='the original text for this point.')

class Note(BaseModel):
    query: str = Field(default='',description='propose a possiable query for this context.')
    summary: str = Field(
        default='', description="a detailed description of this context. you should make sure this description can greatly help reader undestand the original text.")
    reference: str = Field(
        default='', description="list all detailed points of this context and also conclude some possiable deep questions and answers")
    points: List[SubNote] = Field(
        default_factory=list, description="a list of main points of the text")

class Sample(BaseModel):
    doc_id: str = ''
    chunk_id: int = ''
    chunk_ids: str = ''
    chunk_ids_list: List[int] = Field(default_factory=list)
    chunk: str = ''
    cluster_sign: bool = False  # false不是聚类分块
    note: Note = Field(default_factory=Note)

    
if __name__ == "__main__":
    print(Note.model_json_schema())
    note = Note()
    note.points.append(SubNote())
    print(note.model_dump_json())