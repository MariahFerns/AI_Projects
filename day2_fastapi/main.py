from fastapi import FastAPI

from routes import chat

app = FastAPI()
app.include_router(chat.router) # POST 


@app.get("/")                   # GET
def health():
    return {"status": "ok"}

# Start the FastAPI server on port 8000
# python -m uvicorn main:app --port 8000
# The app is running at http://127.0.0.1:8000