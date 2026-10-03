# step LLM mastral with huggingface 
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_PATH = BASE_DIR / "faiss_index"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")


def load_llm():
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2",
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=generator)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {},
    )


def create_vector_store():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"PDF folder not found: {DATA_DIR}")

    loader = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(FAISS_PATH))
    return db


def load_vector_store():
    if FAISS_PATH.exists():
        try:
            return FAISS.load_local(str(FAISS_PATH), get_embeddings(), allow_dangerous_deserialization=True)
        except Exception:
            pass
    return create_vector_store()


def make_prompt():
    template = """Use only the information in the context below to answer the question.
If the answer is not in the context, say that you do not know.
Do not invent facts.

Context:
{context}

Question: {question}
Answer directly, without small talk."""
    return PromptTemplate(input_variables=["context", "question"], template=template)


llm = load_llm()
vectorstore = load_vector_store()
user_query = input("write your query here: ").strip()

retrieved_docs = vectorstore.similarity_search(user_query, k=3)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
formatted_prompt = make_prompt().format(context=context, question=user_query)
response = llm.invoke(formatted_prompt)

print("result:", response)
print("source documents:")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"[{i}] {doc.metadata.get('source', 'unknown')}\n{doc.page_content[:250]}\n")