from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#load row data from pdf file
Data_path="/home/Abbas/Desktop/Medical Chartbot/.venv/data"
def load_pdf_files(Data_path):
    loader = DirectoryLoader(Data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    Document = loader.load()
    return Document
documents = load_pdf_files(Data_path)
#print("Length of documents:", len(documents))
# split the documents into chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_text = text_splitter.split_documents(documents)
    return chunks_text
chunks = create_chunks(documents)
#print("Length of chunks:", len(chunks))

# create vecter enbidings for each chunk
def get_embeddings_model():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model
embeding_model=get_embeddings_model()

# create a vector store for the FAISS 
DB_FAISS_PATH="/home/Abbas/Desktop/Medical Chartbot/.venv/faiss_index"
db=FAISS.from_documents(chunks, embeding_model)
db.save_local(DB_FAISS_PATH)