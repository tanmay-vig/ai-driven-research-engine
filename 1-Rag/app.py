import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    You are Commercial Court AI Assisant Avni and you have give all answer using web and context 
    You the context for your understanding and give answers to the users.

    <context>
    {context}
    <context>
    Question:{input}

    """

)

# def create_vector_embedding():
#     faiss_store_path = "faiss_store"

#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         st.session_state.loader = PyPDFDirectoryLoader("dataset")  # Data Ingestion step
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
#         # Check if FAISS vector store exists
#         if os.path.exists(faiss_store_path):
#             st.session_state.vectors = FAISS.load_local(faiss_store_path, st.session_state.embeddings)
#         else:
#             st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         # Save FAISS vector store locally
#             st.session_state.vectors.save_local(faiss_store_path)    

st.title("Research Engine for Commercial Courts")

user_prompt=st.text_input("Enter your query from the research paper")

# if st.button("Document Embedding"):
#     create_vector_embedding()
#     st.write("Vector Database is ready")

import time

faiss_store_path = "faiss_store"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use the same embeddings as before
vector_store = FAISS.load_local(faiss_store_path, embeddings, allow_dangerous_deserialization=True)



if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vector_store.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

st.write('<a href="mailto:cooltanmayvig@gmail.com"> Reach to Us and Give Feedback !</a>', unsafe_allow_html=True)