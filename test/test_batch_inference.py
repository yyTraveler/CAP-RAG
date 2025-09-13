import os
import sys
sys.path.append(os.path.abspath(""))
import asyncio
from interface.entity import Note
from interface.chat import LLMChat




def test_batch_infer():
    prompts = [
        "你好，我是xxx"
    ]
    notes = asyncio.run(LLMChat.batch_chunk_preparse("Qwen/Qwen2.5-7B-Instruct",prompts))
    print(notes)
    for note in notes:
        if note:
            note: Note
            print(note.model_dump_json(indent=4))
        print("============")



if __name__ == "__main__":
    test_batch_infer()