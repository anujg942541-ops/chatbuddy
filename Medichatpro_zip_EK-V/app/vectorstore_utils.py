from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_faiss_index(chunks):
    """
    Create a FAISS index from text chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}  # force CPU for Colab
    )
    # Build FAISS vectorstore instead of returning raw list
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def retrive_relevant_docs(vectorstore: FAISS, query: str, k: int = 4):
    """
    Retrieve top-k relevant documents from the FAISS index.
    """
    return vectorstore.similarity_search(query, k=k)
