from langchain.callbacks.base import AsyncCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import bs4
import uvicorn
import os
import cv2
import numpy as np
import mediapipe as mp

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_iris_position(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return "No face detected"

    face_landmarks = results.multi_face_landmarks[0]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    eye_iris_x = [0, 0]

    for eye, iris, side in [(LEFT_EYE, LEFT_IRIS, "L"), (RIGHT_EYE, RIGHT_IRIS, "R")]:
        eye_points = [face_landmarks.landmark[i] for i in eye]
        iris_points = [face_landmarks.landmark[i] for i in iris]

        eye_x = (eye_points[0].x + eye_points[1].x) / 2
        iris_x = sum(point.x for point in iris_points) / len(iris_points)

        if side == "L":
            eye_iris_x[0] = eye_x - iris_x
        elif side == "R":
            eye_iris_x[1] = eye_x - iris_x

    avg_x = (eye_iris_x[0] + eye_iris_x[1]) / 2
    # print(avg_x)

    if avg_x < -0.007:
        return "left"
    elif avg_x > 0.007:
        return "right"
    else:
        return "center"
    
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

@app.post("/detect_iris")
async def detect_iris_position(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    position = get_iris_position(image)
    return {"iris_position": position}

#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8000)
