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
