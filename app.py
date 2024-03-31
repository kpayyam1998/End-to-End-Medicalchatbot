from flask import Flask,render_template
from src.helper import download_hugging_face_embedding

import pinecone
from langchain_pinecone.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.llms import OpenAI

from src.prompt import *
from dotenv import load_dotenv
import os

load_dotenv()

app=Flask(__name__)

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

embeddings=download_hugging_face_embedding()

index_name="medical-chat"

# Key connection of Pinecone
pc=pinecone.Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pass

# Loading index
vector_db=Pinecone.from_existing_index(index_name,embeddings)

PROMPT=PromptTemplate(input_variables=["context","question"],template=prompt_template)
chain_type_kwargs={"prompt":PROMPT}

llm=CTransformers(model="Models/llama-2-7b-chat.ggmlv3.q4_0.bin",
              model_type="llama",
              config={
                  'max_new_tokens':512,
                  'temperature':0.8
              })

retriever=vector_db.as_retriever(search_kwargs={'k':2})
# llm=OpenAI(openai_api_key=OPENAI_API_KEY)
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('index.html')


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)