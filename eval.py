from giskard.rag import KnowledgeBase, generate_testset, QATestset
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
import os
from openai import OpenAI
from giskard.llm.client.openai import OpenAIClient
from giskard.llm.embeddings.openai import OpenAIEmbedding
import giskard.llm
from giskard.rag import AgentAnswer
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_faithfulness
from typing import Optional, Sequence
import numpy as np
from giskard.utils.iterables import batched
from giskard.llm.client import get_default_llm_api
from giskard.llm.embeddings.base import BaseEmbedding
from giskard.rag.metrics.ragas_metrics import RagasMetric
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
import requests
from main import conversational_rag_chain
import asyncio
import csv

class OllamaEmbedding(BaseEmbedding):
    def __init__(self, base_url: str, model: str, batch_size: int = 40):
        """
        Parameters
        ----------
        base_url : str
            Base URL for the Ollama API.
        model : str
            Model name.
        batch_size : int, optional
            Batch size for embeddings, by default 40.
        """
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for batch in batched(texts, self.batch_size):
            response = self._get_embeddings(batch)
            embeddings.extend(response)
        return np.array(embeddings)

    def _get_embeddings(self, texts: Sequence[str]) -> list:
        url = f"{self.base_url}/api/embeddings"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        embeddings = []
        for text in texts:
            data = {
                "model": self.model,
                "prompt": text
            }
            response = requests.post(url, headers=headers, json=data, timeout=None)
            response.raise_for_status()
            embeddings.append(response.json()['embedding'])
        return embeddings

def try_get_ollama_embeddings() -> Optional[OllamaEmbedding]:
    try:
        base_url = "http://localhost:11434"  # Default Ollama URL
        return OllamaEmbedding(base_url=base_url, model="nomic-embed-text")  # You can change the default model
    except:
        return None


# Setup the Ollama client with API key and base URL
_client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
oc = OpenAIClient(model="llama3", client=_client)
embed = try_get_ollama_embeddings()
giskard.llm.set_default_client(oc)

mypath = "./uploaded_files"
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
documents = []
for file_path in onlyfiles:
    file_path = os.path.join(mypath, file_path)
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        continue
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_nodes = text_splitter.split_documents(documents)
knowledge_base_df = pd.DataFrame([node.page_content for node in text_nodes], columns=["text"])
knowledge_base = KnowledgeBase(knowledge_base_df, llm_client=oc, embedding_model=embed)

# testset = generate_testset(knowledge_base,
#                            num_questions=120,
#                            agent_description="A chatbot answering questions the solution architecure of promo modules for a software company named carry1st")


# testset.save("testset1.jsonl")

# You can easily load it back
from giskard.rag import QATestset

loaded_testset = QATestset.load("testset1.jsonl")
df = loaded_testset.to_pandas()

from giskard.rag import evaluate
import uuid

# Wrap your RAG model
async def get_answer_fn(question: str, session_id=None) -> str:
    """A function representing your RAG agent."""

    # Get the answer
    result = await conversational_rag_chain.ainvoke(
            {"input": question},
            {"configurable": {"session_id": session_id}},
        )
    answer = result['answer']
    source_documents = result.get('context', [])

    return AgentAnswer(message=answer, documents=[doc.page_content for doc in source_documents])

async def get_answers(df):
    uid = str(uuid.uuid4())
    answers = []
    for i, row in df.iterrows():
        question = row["question"]
        answer = await get_answer_fn(question, uid)
        answers.append(answer)
    return answers

def write_answers_to_csv(df, answers, output_file='answers.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Answer', 'Source Documents'])
        
        for answer in answers:
            writer.writerow([
                answer.message,
                '; '.join(answer.documents)  # Join all documents into a single string
            ])

# Run the asynchronous function
# answers = asyncio.run(get_answers(df))

# write_answers_to_csv(df, answers)

def read_answers_from_csv(file_path='answers.csv'):
    answers = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:  # Ensure we have at least answer and source documents
                answer = row[0]
                documents = row[1].split('; ') if row[1] else []
                answers.append(AgentAnswer(message=answer, documents=documents))
    return answers

# Read the answers from the CSV file
answers = read_answers_from_csv()


# Run the evaluation and get a report
rcp = RagasMetric(name="RAGAS Context Precision", metric=context_precision, requires_context=True, llm_client=oc, embedding_model=embed)
rf = RagasMetric(name="RAGAS Faithfulness", metric=faithfulness, requires_context=True, llm_client=oc, embedding_model=embed)
rar = RagasMetric(name="RAGAS Answer Relevancy", metric=answer_relevancy, requires_context=True, llm_client=oc, embedding_model=embed)
rcr = RagasMetric(name="RAGAS Context Recall", metric=context_recall, requires_context=True, llm_client=oc, embedding_model=embed)
report = evaluate(answers, testset=loaded_testset, knowledge_base=knowledge_base, metrics=[rcr, rf, rar, rcp])
report.to_html("rag_eval_report.html")