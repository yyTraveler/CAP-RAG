import json
import asyncio
from openai import OpenAI
from joblib import Memory
import traceback

from interface.entity import Note, SubNote
from interface.models import get_openai_client
from interface.prompts import preparse_prompt_schema
from config.llm_config import llm_args
from config.cache_config import cache_args

memory = Memory(location=cache_args.cache_dir, verbose=0)

quality_option_index = ["(a)", "(b)", "(c)", "(d)", "(e)"]
sample_note =  Note()
sample_note.points.append(SubNote())


@memory.cache
async def chat_quality_generate(model_name: str, user_prompt: str):
    """针对quality这种选项数据集，和LLM对话拿到LLM做出的选择选项"""
    completion = await get_openai_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You are expected to do multiple choice questions and return the letter you choose, like return (a)"},
            {"role": "user", "content": user_prompt}
        ],
        extra_body={
            "guided_choice": ["(a)", "(b)", "(c)", "(d)", "(e)"]
        },
        max_tokens=30,
        temperature=0.7,
        top_p=0.8
    )
    response = completion.choices[0].message.content
    return response

@memory.cache
async def preparse_chunk(model_name: str, chunk_content: str):
    """预解析chunk"""
    completion = await get_openai_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given context {chunk_content}, read and understand the context above and then parse it in this json format {sample_note.model_dump_json()} ."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "note",
                "schema": Note.model_json_schema()
            }
        },
        max_tokens=1024,
        temperature=0.7,
        top_p=0.8
    )
    response = completion.choices[0].message.content
    note = None
    try:
        note = Note.model_validate_json(response)
    except Exception as e:
        traceback.print_exception(e)
    return note

@memory.cache
async def summarize_chunk(model_name: str, context: str):
    """对chunk进行总结"""
    completion = await get_openai_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": context}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512
    )
    response = completion.choices[0].message.content
    return response


class LLMChat:
    
    @staticmethod
    async def batch_chunk_preparse(model_name: str, chunk_contents: list[str]):
        """批量预解析"""
        return await asyncio.gather(*(preparse_chunk(model_name, c) for c in chunk_contents))

    @staticmethod
    async def quality_chat(model_name: str, context: str, query: str, options: list[str]) -> tuple[int, str]:
        """针对quality这种选项数据集，和LLM对话拿到LLM做出的选择选项"""
        str_options = '\n'.join([
            f" {quality_option_index[index]}: " + each
            for index, each in enumerate(options + ["UNKNOWN"])
        ])

        prompt = (
            f"Given the following context: {context}. "
            f"Based on the context, answer this single choice question. \n"
            f"The question is : {query} \n"
            f"the options are as follows: {str_options} "
        )
        response = await chat_quality_generate(model_name, prompt)

        if not response:
            return -1, 'no response'
        try:
            answer_index = quality_option_index.index(response)
            return answer_index + 1, ''
        except ValueError:
            return -1, f"value error in response: {response}"
        except:
            return -1, f'response load error: {response}'

    @staticmethod
    async def batch_quality_chat(model_name: str, contexts: list[list], queries: list[str], options: list[list[str]]) -> list[tuple[int, str]]:
        return await asyncio.gather(
            *(LLMChat.quality_chat(model_name, "...\n\n<new chunk start here> ".join(c), q, o) for c, q, o in zip(contexts, queries, options))
        )
    
    @staticmethod
    async def batch_summary(model_name: str, contexts: list[str]):
        return await asyncio.gather(
            *(summarize_chunk(model_name, c) for c in contexts)
        )
    
    