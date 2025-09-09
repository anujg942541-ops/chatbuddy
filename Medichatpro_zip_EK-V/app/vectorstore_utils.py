#Two operation need to perform one is store the data and retriev the data. 

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

#Store data in the form of embeddings 
def create_faiss_index(texts: List[str]) :
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts, embeddings)

#Top 4 results 
def retrive_relevant_docs(vectorstore: FAISS, query: str, k: int = 4):
    return vectorstore.similarity_search(query, k=k)


