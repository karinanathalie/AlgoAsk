# This script performs question answering via RAG
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path

def process_uploaded_file(file):
    df = pd.read_csv(file)
    texts = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(texts, embeddings, persist_directory="embeddings")

def load_qa_chain():
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="embeddings", embedding_function=embeddings)
    retriever = db.as_retriever()
    # Load LLM
    llm = LlamaCpp(
        model_path=str(Path("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()),
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        n_gpu_layers=0,
        use_mlock=False,
        verbose=True,
    )
    # Use Mistral-style prompt
    prompt_template = """[INST] You are an expert trading analyst. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    # Return RetrievalQA with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def ask_question(question):
    qa = load_qa_chain()
    answer = qa.run(question)
    return answer

if __name__ == "__main__":
    print(ask_question("What is the average slippage for VWAP?"))