
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rag_core import get_answer
from fastapi.responses import FileResponse
app = FastAPI()

# Enable CORS for frontend JS
app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)


app.mount("/public", StaticFiles(directory="public", html=True), name="public")
@app.get("/")
async def serve_index():
    return FileResponse("public/index.html")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    return get_answer(query)
