from langchain.callbacks.base import AsyncCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import bs4
import uvicorn
import os

class LLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_json({"message": token})

os.environ["LANGCHAIN_API_KEY"] = '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'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = JSONLoader(
    file_path='roadmapss.json',
    jq_schema='.',
    text_content=False
)
docs = loader.load()

embed_model = OllamaEmbeddings(base_url='http://localhost:11434',
                               model="mxbai-embed-large")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

# Ensure that you handle cases where there may be fewer documents than requested
if len(splits) > 0:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
    retriever = vectorstore.as_retriever()
else:
    raise ValueError("No documents found after splitting.")

prompt = hub.pull("rlm/rag-prompt")


@app.get("/predict")
def predict(query: str, lang: str, role: str):
    llm = ChatOllama(
        base_url='http://localhost:11434',
        model="llama3.1:8b",
        temperature=0,
        format="json"
    )

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
    )

    messages = [
        ("system", "You are an HR professional working in an information technology (IT) company. A user has applied for a position as {role} in your company. Always use {language} as the main response."),
        ("human", "{input}. Respond using JSON only."),
    ]

    language = "bahasa Indonesia" if lang == "id" else "English"

    response = rag_chain.invoke(
        {
            "role": role,
            "language": language,
            "input": query,
        }
    )

    return response


#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8000)
