from fastapi import APIRouter
from pydantic import BaseModel

from services import ollama_service

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


# Plain def so the blocking requests call runs in FastAPI's threadpool
@router.post("/chat")
def chat(req: ChatRequest):
    answer = ollama_service.ask(req.question)
    return {"answer": answer}
