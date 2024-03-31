import ray
import os
import psycopg
from dotenv import load_dotenv
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ray.data import ActorPoolStrategy
from pgvector.psycopg import register_vector

DB_CONNECTION_STRING = "dbname=postgres user=postgres host=pgvector password=postgres"

ray.init()

# Path for Ray docs.
ray_docs_path = Path("ray-assistant-data/docs.ray.io/en/master/").resolve()
assert ray_docs_path.exists()

# Load data
ds = ray.data.from_items([{"path": path} for path in ray_docs_path.rglob("*.html") if not path.is_dir()])
print(f"{ds.count()} documents")

# Utility functions to extract sections from a Ray document.
def extract_sections(record):
    with open(record["path"], "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    sections = soup.find_all("section")
    section_list = []
    for section in sections:
        section_id = section.get("id")
        section_text = extract_text_from_section(section)
        if section_id:
            uri = path_to_uri(path=record["path"])
            section_list.append({"source": f"{uri}#{section_id}", "text": section_text})
    return section_list

def extract_text_from_section(section):
    texts = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            if elem.strip():
                texts.append(elem.strip())
        elif elem.name == "section":
            continue
        else:
            texts.append(elem.get_text().strip())
    return "\n".join(texts)

def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]

# Extract sections
sections_ds = ds.flat_map(extract_sections)
section_lengths = []
for section in sections_ds.take_all():
    section_lengths.append(len(section["text"]))

# Chunk sections
chunk_size = 300
chunk_overlap = 50

def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.create_documents(
        texts=[section["text"]],
        metadatas=[{"source": section["source"]}])
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
chunks_ds = sections_ds.flat_map(partial(
    chunk_section,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap))

# Generate embeddings
def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # also works with model_path
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    return embedding_model

class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100})
    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}

embedding_model_name = "thenlper/gte-base"
embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=100, 
    num_gpus=1,
    compute=ActorPoolStrategy(size=1))

# Index data
class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(DB_CONNECTION_STRING) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                    cur.execute("INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)", (text, source, embedding,),)
        return {}

# Initialize Postgres
with psycopg.connect(DB_CONNECTION_STRING) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION vector;")
        cur.execute("CREATE TABLE document (id serial primary key, text text not null, source text not null, embedding vector(768));")

embedded_chunks.map_batches(
    StoreResults,
    batch_size=128,
    num_cpus=1,
    compute=ActorPoolStrategy(size=6),
).materialize()
