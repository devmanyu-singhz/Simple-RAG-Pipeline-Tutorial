from src.rag.data_loader import load_all_docs
from src.rag.embedding import EmbeddingPipeline
from src.rag.vectorstore import FaissVectorStore

# example usage

if __name__ == "__main__":
    # docs = load_all_docs("data")          the commented codes are used first to create the faiss store
    store=FaissVectorStore("faiss_store")
    # store.build_from_docs(docs)
    store.load()
    print(store.query("what is reverse recovery time", top_k=5))

# in the search folder we can integrate LLMs