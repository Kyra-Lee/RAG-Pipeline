# Retrieval-Augmented Generation (RAG) Pipeline: Setup and Usage Guide

This document outlines a complete procedure for building a local RAG pipeline using Python. It covers environment setup, installation of tools, document cleaning, vector embedding, and prompt generation for use with a local LLM frontend such as LM Studio.

---

## 1. Environment Setup

### A. Project Directory Structure
Create a directory for your project and organize it as follows:

```
project-folder/
├── raw-html/              # Raw HTML documents
├── cleaned-html/          # HTML after tag-based cleanup
├── cleaned-markdown/      # Markdown output for embedding
├── chromadb-index/        # Persistent vector database
├── parse_html.py          # HTML cleaning + markdown generation
├── embed_documents.py     # Embedding script
├── query_with_context.py  # (Optional) Prompt generation script
├── .venv/                 # Virtual environment
```

### B. Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 2. Install Required Packages

```bash
pip install -U \
  beautifulsoup4 \
  unstructured[html] \
  langchain \
  langchain-community \
  chromadb \
  sentence-transformers \
  tiktoken
```

---

## 3. Clean and Convert HTML to Markdown (`parse_html.py`)

This script cleans raw HTML and converts it into readable markdown using `BeautifulSoup` and `unstructured`.

```python
import os
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html

def clean_html_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Retain only the main content area
    content_root = soup.find("div", id="mw-content-text")
    if content_root:
        soup.body.clear()
        soup.body.append(content_root)

    # Remove UI clutter
    for selector in [
        ".navbox", ".printfooter", ".toc", ".mw-editsection",
        ".mw-parser-output .hlist"
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Remove junk strings
    for el in soup.find_all(string=lambda s: (
        "wiki." in s or
        "Retrieved from" in s or
        "In other languages:" in s
    )):
        el.extract()

    # Remove the entire "See also" section if it exists
    see_also_span = soup.find("span", id="See_also")
    if see_also_span:
        heading = see_also_span.find_parent("h2")
        if heading:
            current = heading
            while current:
                next_node = current.find_next_sibling()
                if next_node and next_node.name and next_node.name.startswith("h"):
                    break
                current.decompose()
                current = next_node
            heading.decompose()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

# Main script
input_dir = "raw-html"
temp_dir = "cleaned-html"
output_dir = "cleaned-markdown"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        raw_path = os.path.join(input_dir, filename)
        cleaned_path = os.path.join(temp_dir, filename)
        clean_html_file(raw_path, cleaned_path)

        # Convert cleaned HTML to markdown
        elements = partition_html(filename=cleaned_path)
        content = "\n\n".join([el.text for el in elements])

        out_path = os.path.join(output_dir, filename.replace(".html", ".md"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

print("✅ HTML cleaned and converted to markdown.")
```

---

## 4. Embed Documents in ChromaDB (`embed_documents.py`)

This script converts your markdown into vector embeddings and stores them in a Chroma database.

```python
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
```

---

## 5. Optional: Generate Context-Enhanced Prompts (`query_with_context.py`)

This script allows you to query the Chroma index and copy the top-k results for pasting into an LLM like LM Studio.

```python
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Config
persist_dir = "chromadb-index"
embedding_model = "all-MiniLM-L6-v2"
top_k = 5

embedding = HuggingFaceEmbeddings(model_name=embedding_model)
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

query = input("🔍 Enter your question: ")
results = vectordb.similarity_search(query, k=top_k)

prompt = "You are an expert. Use the following information to answer the user's question.\n\n"
for i, doc in enumerate(results):
    prompt += f"[Source {i+1}]\n{doc.page_content}\n\n"
prompt += f"User's question: {query}\n\nAnswer:"

print("\n📋 Copy this prompt into your LLM chat:")
print("======== COPY BELOW ========")
print(prompt)
print("======== COPY ABOVE ========")
```

---

## 6. What's Next?

You can evolve this into a fully automated system using:
- FastAPI or Flask for a local RAG API
- Open WebUI as a chat frontend
- llama-cpp-python for integrated local LLM inference

This setup ensures high-quality, local document understanding with maximum control.
