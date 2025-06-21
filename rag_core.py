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
collection_name = "chai_code_docs_final"

qdrant_client = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key
)

# Now have to initialize the LangChain wrapper
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model
)


def rewrite_in_hitesh_persona(query: str, initial_answer: str) -> str:
    """
    Rewrites the assistant's response in Hitesh Sir's teaching tone.
    """
    try:
        persona_prompt = f"""
        Haanji! Kaise ho aap? 👋

        You are now speaking like *Hitesh Sir*, a beloved Indian educator known for his warm energy, practical coding tutorials, and humorous analogies.
        Your goal is to help students clearly understand a concept with real life examples as if you're teaching it in a YouTube video or live session.

        Tone Guidelines:
        - Start with "Haanji! Kaise ho aap? 👋" to make it friendly.
        - Use analogies to explain complex ideas in a simple way.
        - Use phrases like:
          - "Code karna aasan hai, par topic ko samajh loge to..."
          - "Chai pite raho aur questions puchte raho!"
          - "Ek kaam karo..."
        - Be supportive, slightly humorous, and very clear.
        - Address the learner directly as if you’re in a live class.

        Examples:
        Don’t say: “To authenticate, use the following method.”
        Do say: “Ek kaam karo – `supabase.auth.signIn()` use karo. Ye bilkul waise hi hai jaise tum Instagram login karte ho.”

        Original Query: {query}

        Initial Answer: {initial_answer}

        Rewrite the answer in Hitesh Sir’s tone based on the above.
        Only return the improved answer, nothing else.
        """

        messages = [
            {"role": "system", "content": persona_prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.warning("Persona rewriting failed: %s", str(e))
        return initial_answer


def get_answer(query: str) -> dict:
    """
    Returns an answer to the query using vector search and persona rephrasing.
    """
    try:
        search_results = vector_db.similarity_search(query=query)

        if not search_results:
            return {
                "content": "❌ I couldn't find any relevant documents to answer your query.",
                "url": None
            }

        context = "\n\n\n".join(
            [f"Page Content: {r.page_content}" for r in search_results]
        )

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on technical documentation content retrieved from various web pages.
        Only answer based on the context below and help guide the user by referencing the relevant documentation source URLs.

        Respond strictly in the following JSON format:
        {{
            "content": "Your helpful answer goes here with references to page numbers or sources."
        }}

        Dont provide "Source:" because it is getting appended in the final answer!

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
        base_answer = parsed.get("content", "Sorry, I couldn't find an answer.")

        # Rewrite the answer in Hitesh Sir's tone
        final_answer = rewrite_in_hitesh_persona(query, base_answer)
        unique_urls = list({r.metadata.get('source', 'unknown') for r in search_results})

        if unique_urls:
            source_links_text = "<br>".join([f'<a class="source-link" href="{url}" target="_blank">📄 {url}</a>' for url in unique_urls])
            final_answer += f"\n\nSources:\n{source_links_text}"

        return {
            "content": final_answer,
            "urls": unique_urls
        }

    except Exception as e:
        logging.error("Error in get_answer: %s", str(e))
        return {
            "content": f"❌ Error: {str(e)}",
            "url": None
        }
