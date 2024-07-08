from typing import List
import os

from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.base import BaseLLMOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama



# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(BaseLLMOutputParser[LineList]):
    def __init__(self) -> None:
        super()

    def parse_result(self, text: str) -> LineList:
        text = text[0].text
        lines = text.strip().split("\n")
        lines = [line for line in lines if "?" in line]
        return LineList(lines=lines)


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://localhost:11434')
llm = ChatOllama(base_url=OLLAMA_BASE, model="llama3", temperature=0)

mqr_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
