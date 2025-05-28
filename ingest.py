# This script loads CSVs and turns them into vector embeddings
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from config import OPENAI_API_KEY

def load_csv(filepath):
    df = pd.read_csv(filepath)
    texts = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
    return texts

def build_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory="embeddings")
    db.persist()
    print("Vectorstore built and saved locally.")

if __name__ == "__main__":
    data = load_csv("data/sample_report.csv")
    build_vectorstore(data)