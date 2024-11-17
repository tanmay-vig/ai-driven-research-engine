import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv
load_dotenv()



## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm=Ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## #Title of the app
st.title("Enhanced Q&A Research Engine with ModelFile")


## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",[ "major", "llama3.2"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
st.sidebar.markdown('<a href="mailto:cooltanmayvig@gmail.com"> Reach to Us and Give Feedback !</a>', unsafe_allow_html=True)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

submit=st.button("Ask the question")


if user_input and submit :
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")


