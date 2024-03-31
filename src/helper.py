from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings




# Extract data from pdf
def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


# Extract data to text chunks
def text_split(extracted_data):
    text_spilitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_spilitter.split_documents(extracted_data)
    return text_chunks

# Embeddings

def download_hugging_face_embedding():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    return embeddings