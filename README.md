# End-to-End-Medicalchatbot

Steps:

Create one environment:

    conda create -n chatbot python=3.8 -y

Install

    note:
        please mentions the version of the Lib
    
    pip install -r requirements.txt  

    Download : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML model

Notebook:
    
    Create onebook just for implementation

Model link 

    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
    Download and put it in your local folder

PINECONE:
    
    create pinecone api key and index 
    https://www.pinecone.io/

Project Implementation 

    1. Load the data such  as text,.docx,.pdf etc....
    
    2. Create chunks our data so that we can pass to the model 

    3. Split data and Get Chunks

    4. Embedding

        During pinecone vectorDB time you may face error. that  you have to import

            from langchain_pinecone.vectorstores import Pinecone as pc_index

            or else gothrough Langchain from_text 

            https://python.langchain.com/docs/integrations/retrievers/self_query/pinecone#creating-a-pinecone-index

            This is the page i referred to solve my problem

Note :
    Dont forget to set Environment variables and restart your machine

    set PINECONE_API_KEY 
    set OPENAI_API_KEY

    you may get error When you try to connect Pinecone. please gothrought their documentation before connect DB


