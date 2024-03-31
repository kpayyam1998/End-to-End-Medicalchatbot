from src.helper import load_pdf,text_split,download_hugging_face_embedding
import pinecone
from langchain_pinecone.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
# print(PINECONE_API_KEY)


extracted_data=load_pdf("Data/")

text_chunks=text_split(extracted_data)

embeddings=download_hugging_face_embedding() 


index_name="medical-chat"

# Key connection of Pinecone
pc=pinecone.Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pass

vector_db=Pinecone.from_documents(text_chunks,embeddings,index_name=index_name)

#vector_db=Pinecone.from_text([t.page_content for t in text_chunks],embeddings,index_name=index_name)






