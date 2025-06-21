# scrape_and_index.py

from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

BASE_URL = "https://docs.chaicode.com"
START_PAGE = "https://docs.chaicode.com/youtube/getting-started/"

def collect_sidebar_links(start_page):
    """Scrape all sidebar/internal links starting from a known page."""
    print(f"🔍 Scraping links from: {start_page}")
    try:
        response = requests.get(start_page, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to load page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/youtube/") and not href.endswith("/youtube/"):
            full_url = BASE_URL + href
            links.add(full_url)

    # print(f"Found {len(links)} unique documentation pages.")
    return sorted(links)

urls = collect_sidebar_links(START_PAGE)

loader = WebBaseLoader(urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
split_docs = text_splitter.split_documents(documents=docs)

for doc in split_docs:
    doc.metadata["source"] = doc.metadata.get('source', 'unknown').split("#")[0]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

@retry(wait=wait_exponential(min=4, max=10), stop=stop_after_attempt(3))
def index_documents():
    print("Indexing documents to Qdrant...")
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"), 
        collection_name="chai_code_docs_final",
        embedding=embedding_model,
        batch_size=32
    )
    print("Indexing done.")

index_documents()
