from typing import Union
import json
import os
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
client = OpenAI()

# Qdrant client for collection check 
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "chai_code_docs"

qdrant_client = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key
)

# Have to Create the collection if it doesn't exist
if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

# Now have to initialize the LangChain wrapper
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model
)

def get_answer(query: str) -> dict:
    try:
        search_results = vector_db.similarity_search(query=query)

        if not search_results:
            return {
                "content": "❌ I couldn't find any relevant documents to answer your query.",
                "url": None
            }

        context = "\n\n\n".join(
            [f"Page Content: {r.page_content}\nSource URL: {r.metadata.get('source', 'unknown')}" for r in search_results]
        )

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on technical documentation content retrieved from various web pages.
        Only answer based on the context below and help guide the user by referencing the relevant documentation source URLs.

        Respond strictly in the following JSON format:
        {{
            "content": "Your helpful answer goes here with references to page numbers or sources."
        }}

        Context:
        {context}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(response.choices[0].message.content)
        content = parsed.get("content", "Sorry, I couldn't find an answer.")

        match = re.search(r"Source URL: (https?://[^\s]+)", context)
        source_url = match.group(1) if match else None

        return {
            "content": content,
            "url": source_url
        }

    except Exception as e:
        logging.error("Error in get_answer: %s", str(e))
        return {
            "content": f"❌ Error: {str(e)}",
            "url": None
        }
