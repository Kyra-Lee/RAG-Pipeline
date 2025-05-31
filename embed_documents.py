import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
docs_dir = "cleaned-markdown"
persist_dir = "chromadb-index"
embedding_model = "all-MiniLM-L6-v2"

# Load markdown files
docs = []
for filename in os.listdir(docs_dir):
    if filename.endswith(".md"):
        loader = TextLoader(os.path.join(docs_dir, filename), encoding="utf-8")
        docs.extend(loader.load())

# Chunk the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
vectordb.persist()
print("✅ Documents embedded and saved to ChromaDB.")
