# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from rag_core import get_answer

# app = FastAPI()

# # Allow CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or restrict to ['http://localhost:8000']
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve static HTML files
# app.mount("/public", StaticFiles(directory="public", html=True), name="static")

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/chat")
# async def chat_endpoint(request: QueryRequest):
#     answer = get_answer(request.query)
#     return {"response": answer}

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rag_core import get_answer

app = FastAPI()

# Enable CORS for frontend JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static HTML/JS/CSS from /public
app.mount("/", StaticFiles(directory="public", html=True), name="public")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    return get_answer(query)
