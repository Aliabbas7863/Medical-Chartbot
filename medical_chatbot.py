import streamlit as st
import os
from transformers import pipeline
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()  # Load environment variables from .env file
# Optional imports from langchain; provide graceful fallback if unavailable
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except Exception:
    HuggingFaceEmbeddings = None
    HuggingFaceEndpoint = None
    RetrievalQA = None
    FAISS = None
    PromptTemplate = None
    HAS_LANGCHAIN = False

DB_FAISS_PATH = "/home/Abbas/Desktop/Medical Chartbot/.venv/faiss_index"


@st.cache_resource
def get_vectorstore():
    if not HAS_LANGCHAIN:
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


@st.cache_resource
def build_faiss_from_pdf(pdf_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # Extract text from PDF, split into chunks, embed, and build FAISS index
    texts = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # create sliding chunks of ~500 chars with 100 char overlap for better recall
                    raw = page_text.replace('\r', '\n')
                    raw = "\n\n".join([p.strip() for p in raw.split('\n\n') if p.strip()])
                    chunk_size = 500
                    overlap = 100
                    start = 0
                    while start < len(raw):
                        chunk = raw[start:start+chunk_size].strip()
                        if len(chunk) > 30:
                            texts.append(chunk)
                        start += chunk_size - overlap
    except Exception:
        return None, None, None

    if not texts:
        return None, None, None

    embed_model = load_embedding_model(model_name)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    # ensure float32
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, texts, embed_model


def retrieve_from_index(index, texts, embed_model, query: str, top_k: int = 10, threshold: float = 0.35):
    if index is None or texts is None:
        return None
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    scores = D[0]
    ids = I[0]
    results = []
    for score, idx in zip(scores, ids):
        if idx < 0:
            continue
        if score >= threshold:
            results.append((float(score), texts[idx]))
    if not results:
        # if nothing passes threshold, still return the single best hit to improve recall
        # but only if the top score is non-negative
        if len(scores) > 0 and scores[0] > 0:
            best_idx = int(ids[0])
            if best_idx >= 0:
                return texts[best_idx]
        return None
    # return concatenated top results
    return "\n\n".join([r[1] for r in results])


def get_custom_prompt(custom_prompt_template):
    if not HAS_LANGCHAIN:
        return None
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(Huggingface_repo_id=None, HF_TOKEN=None):
    if not HAS_LANGCHAIN:
        # fallback: simple transformers pipeline (tiny model)
        try:
            gen = pipeline('text-generation', model='sshleifer/tiny-gpt2')
            return lambda q: gen(q, max_length=64)[0]['generated_text']
        except Exception:
            return lambda q: "LangChain not available; install langchain to enable QA features."

    llm = HuggingFaceEndpoint(repo_id=Huggingface_repo_id, tamperature=0.5, model_kwargs={"token": HF_TOKEN, "max_new_tokens": 512})
    return llm


def main():
    st.title("Ask Chatbot!")
    if 'message' not in st.session_state:
        st.session_state['message'] = []
    for message in st.session_state['message']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.text_input("Type your query here:")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.message.append({"role": "user", "content": prompt})
        custom_prompt_template = """
                use the pieces of information provided in the context to answer to user's question if 
                the context is helpful. If the context is not helpful, say "I don't know the answer."    If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't provide anything out of the given context

                Context: {context}
                Question: {question}
                Start the answeer direectly.No small talk please."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # If langchain is not available, use local PDF-based retrieval (deterministic)
        if not HAS_LANGCHAIN:
            pdf_path = os.path.join(os.path.dirname(__file__), 'data', 'The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf')
            if not os.path.exists(pdf_path):
                # fallback: try .venv/data
                pdf_path = "/home/Abbas/Desktop/Medical Chartbot/.venv/data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

            index, texts, embed_model = build_faiss_from_pdf(pdf_path, model_name)
            if index is None:
                reply = "No indexed documents found. Place your PDFs in the data/ folder or install LangChain for hosted QA."
                st.chat_message('assistant').markdown(reply)
                st.session_state.message.append({"role": "assistant", "content": reply})
                return

            retrieved = retrieve_from_index(index, texts, embed_model, prompt, top_k=5, threshold=0.55)
            if retrieved is not None:
                # Provide deterministic answer using only retrieved context
                reply = f"Answer based on provided documents:\n\n{retrieved}"
            else:
                reply = "I don't know the answer based on the provided documents."

            st.chat_message('assistant').markdown(reply)
            st.session_state.message.append({"role": "assistant", "content": reply})
            return

        # LangChain path
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vectorstore is not loaded. Please check the FAISS index path.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(Huggingface_repo_id=model_name, HF_TOKEN=os.environ.get('HF_TOKEN')),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt_template}
            )
            response = qa_chain.invoke({"query": prompt})
            result = response['result']
            source_documents = response['source_documents']
            result_to_show = result + "\n\n" + "Source Documents:\n"  + str(source_documents)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.message.append({"role": "assistant", "content": result_to_show})
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
