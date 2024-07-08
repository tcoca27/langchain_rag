import os
import asyncio
import aiofiles
import mimetypes
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

import constants
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
from mqr import mqr_chain
import lancedb.rerankers
import pyarrow as pa

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

reranker = lancedb.rerankers.ColbertReranker()

app = FastAPI()

OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://localhost:11434')
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize LLM and embedding model
llm = ChatOllama(base_url=OLLAMA_BASE, model="gemma2")
embed_model = OllamaEmbeddings(base_url=OLLAMA_BASE, model="nomic-embed-text")

# Initialize vector store
vectorstore = Chroma(embedding_function=embed_model, persist_directory="./chroma_db_id")
search_args = {
    "k": 20,
}
retriever = vectorstore.as_retriever(search_kwargs=search_args)
multi_query_retriever = MultiQueryRetriever(
    retriever=retriever,
    llm_chain=mqr_chain,
    parser_key="lines",
)


def get_neighboring_chunks(doc: Document, k: int = 1) -> List[Document]:
    doc_id = doc.metadata.get('doc_id')
    found = vectorstore.get(where={"doc_id": {"$eq": f"{int(doc_id)+1}"}})
    return [Document(found['documents'][0], metadata=found['metadatas'][0])] if found and len(found['documents']) > 0 else []

def rerank_documents(query: str, documents: List[Document], threshold: float = 0.1, k_neighbors: int = 1) -> List[Document]:
    doc_texts = [doc.page_content for doc in documents]
    k = min(5, len(doc_texts))
    res_rerank = reranker._rerank(query=query, result_set=pa.table([pa.array(doc_texts), pa.array([i for i in range(len(doc_texts))])], names=['text', 'index'])).to_pandas()
    indexes = []
    for i, row in res_rerank.iterrows():
        if row['_relevance_score'] > threshold and len(indexes) < k:
            indexes.append(row['index'])
    reranked_docs = [documents[i] for i in indexes]
    
    # Get neighboring chunks for each reranked document
    expanded_docs = []
    seen_content = set()
    for doc in reranked_docs:
        if doc.page_content not in seen_content:
            expanded_docs.append(doc)
        neighbors = get_neighboring_chunks(doc, k=k_neighbors)
        for neighbor in neighbors:
            seen_content.add(neighbor.page_content)
            if neighbor.page_content not in seen_content:
                expanded_docs.append(neighbor)

    
    print(f"Number of documents after reranking and neighbour search: {len(expanded_docs)}")
    return expanded_docs

class RerankingRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def get_relevant_documents(self, query):
        docs = self.base_retriever.invoke(query)
        print(f"Number of documents retrieved: {len(docs)}")
        # Remove duplicates while preserving order
        unique_docs = []
        seen_content = set()
        for doc in docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        print(f"Number of documents after deduplication: {len(unique_docs)}")
        
        return rerank_documents(query, unique_docs)

retriever = RerankingRetriever(multi_query_retriever)

# Contextualize question
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
qa_system_prompt = """You are an AI assistant for a tech company with the following description: \
Develops, licenses and publishes mobile games in Africa and runs a payments platform that enables in-app purchases across the continent for such popular games. Thats significant since local payment methods in Africa dominate over the card payments which reign supreme in North America. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Your response should be clear and concise, but most importantly, complete for a technical person. Your answer must be based on the provided context, don't add things that aren't there. \
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class DocumentsResponse(BaseModel):    
    text: str
    source_name: Optional[str] = None
    source_path: Optional[str] = None
    page: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[DocumentsResponse]] = None

async def save_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return file_path

def is_valid_file(file: UploadFile) -> bool:
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in constants.ALLOWED_EXTENSIONS:
        return False
    
    if file.size > constants.MAX_FILE_SIZE:
        return False
    
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type not in ['text/plain', 'application/pdf', 'application/msword', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                         'text/csv']:
        return False
    
    return True

@app.get("/upload", response_model=List[str])
async def uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    return files

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    valid_files = [file for file in files if is_valid_file(file)]
    if not valid_files:
        raise HTTPException(status_code=400, detail="No valid files were uploaded")

    try:
        file_paths = await asyncio.gather(*[save_file(file) for file in valid_files])
        
        documents = []
        for file_path in file_paths:
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
        splits = text_splitter.split_documents(documents)

        vectorstore.add_documents(splits)
        vectorstore.persist()

        return JSONResponse(
            content={
                "message": f"{len(valid_files)} file(s) uploaded and indexed successfully",
                "files": [file.filename for file in valid_files],
                "skipped_files": [file.filename for file in files if file not in valid_files]
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return JSONResponse(
            content={"message": "An error occurred while processing the files"},
            status_code=500
        )
    finally:
        for file in files:
            await file.close()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    query = request.message
    session_id = request.session_id
    
    result = conversational_rag_chain.invoke(
        {"input": query},
        {"configurable": {"session_id": session_id}}
    )
    
    response = result['answer']
    for doc in result.get('context', []):
        print(doc)
    print(result)
    source_documents = result.get('context', [])

    sources = []
    for doc in source_documents:
        sources.append(DocumentsResponse(
            text=doc.page_content,
            source_name=doc.metadata.get('source'),
            source_path=doc.metadata.get('file_path'),
            page=doc.metadata.get('page')
        ))

    return ChatResponse(response=response, sources=sources if sources else None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)