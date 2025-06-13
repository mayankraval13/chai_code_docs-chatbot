# flake8: noqa

from dotenv import load_dotenv

from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

# loader = WebBaseLoader("https://docs.chaicode.com/youtube/getting-started/")
# docs = loader.load()

urls = [
    "https://docs.chaicode.com/youtube/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",
    "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
    "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
    "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
    "https://docs.chaicode.com/youtube/chai-aur-django/models/",
    "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
]
docs = []

loader = WebBaseLoader(urls)
docs.extend(loader.load())

# pdf_path = Path(__file__).parent / "nodejs.pdf"

# # Loading
# loader = PyPDFLoader(file_path=pdf_path)
# docs = loader.load() ### Reads the file ###

# Chunking
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=400
)

split_docs = text_splitter.split_documents(documents=docs)
for doc in split_docs:
    doc.metadata["source"] = doc.metadata.get("source", "unknown")
# Vector Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


# Using embedding_model create embeddings of split_docs and store in DB
# Retry upserts in case of failure
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def index_documents():
    print("Indexing to Qdrant...")
    QdrantVectorStore.from_documents(
        documents=split_docs,
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),  # Optional for local Qdrant
        collection_name="chai_code_docs",
        embedding=embedding_model,
        batch_size=32  # Prevent timeout
    )
    print("Indexing of documents DONE.")

# Run indexing
index_documents()