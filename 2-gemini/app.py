import os
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from streamlit_option_menu import option_menu
import PyPDF2  # For extracting text from PDFs

load_dotenv()

st.set_page_config(
    page_title="Court Engine",
    page_icon=":judge:",
    layout="centered",
)

Symbol = "symbol.jpg"
Supreme_Court_Logo = "supreme_court.png"

# Sidebar for navigation with buttons
st.sidebar.title("Navigation")
with st.sidebar:
    selected = option_menu("Main Menu", ["Chatbot", "Upload Doc", "Research" ],  default_index=0)

st.sidebar.markdown('<a href="mailto:cooltanmayvig@gmail.com"> Reach to Us and Give Feedback !</a>', unsafe_allow_html=True)

# Page state control (to maintain the current page across interactions)
if "current_page" not in st.session_state:
    st.session_state.current_page = "Main Chatbot"  # Default page is the Main Chatbot

print(selected)
# Change page when buttons are clicked
if selected=="Chatbot":
    st.session_state.current_page = "Main Chatbot"

if selected=="Upload Doc":
    st.session_state.current_page = "Upload Doc"
if selected =="Research":
    st.session_state.current_page = "Research"


# Google API setup and configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Function to display assistant's response with typing effect
def writeAssistantResponse(response):
    with st.chat_message("assistant", avatar=Symbol):
        placeholder = st.empty()
        typed_message = ""
        for char in response:
            typed_message += char
            placeholder.markdown(typed_message)
            time.sleep(0.05)  # Simulate typing effect


def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

print("Page",st.session_state.current_page)
# PAGE 1: Main Chatbot
if st.session_state.current_page == "Main Chatbot":
    st.image(Supreme_Court_Logo, width=100)
    st.title("Main Court Engine Chatbot")

    # Load structured instructions for the main chatbot
    with open('instructions2.txt', 'r') as file:
        STRUCTURED_INSTRUCTIONS = file.read()

    # Main chatbot model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        system_instruction=STRUCTURED_INSTRUCTIONS
    )

    # Maintain session for the main chatbot
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    for message in st.session_state.chat_session.history:
        text = message.parts[0].text.strip() if message.parts and message.parts[0].text else ""
        if text:
            with st.chat_message(translate_role_for_streamlit(message.role), avatar=Symbol if message.role == "model" else None):
                st.markdown(text)

    user_prompt = st.chat_input("How Can I Help You?")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        loading = st.empty()

        loading.chat_message("assistant", avatar=Symbol).markdown("Processing...")
        response = st.session_state.chat_session.send_message(user_prompt)
        loading.empty()
        writeAssistantResponse(response.text)

# PAGE 2: PDF-based Chatbot

if st.session_state.current_page == "Upload Doc":
    st.image(Supreme_Court_Logo, width=100)
    st.title("PDF-based Chatbot")

    uploaded_pdf = st.file_uploader("Upload a PDF to ask questions", type="pdf")

    if uploaded_pdf:
        # Extract text from the uploaded PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Display extracted text (for verification)
        #st.text_area("Extracted Text from PDF", pdf_text, height=300)

        # Load structured instructions for the PDF-based chatbot
        with open('instructions3.txt', 'r') as file:
            SYSTEM_INSTRUCTIONS_PDF = file.read()

        # Create a model for the PDF-based chatbot
        pdf_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=SYSTEM_INSTRUCTIONS_PDF
        )

        # Maintain session for PDF-based chatbot
        if "pdf_chat_session" not in st.session_state:
            st.session_state.pdf_chat_session = pdf_model.start_chat(history=[])
            st.session_state.pdf_chat_session.send_message("Summarize the following file data : "+pdf_text)
        
        i =0
        for message in st.session_state.pdf_chat_session.history:
            if i>1:
                text = message.parts[0].text.strip() if message.parts and message.parts[0].text else ""
                if text:
                    with st.chat_message(translate_role_for_streamlit(message.role), avatar=Symbol if message.role == "model" else None):
                        st.markdown(text)
            i+=1

        user_pdf_prompt = st.chat_input("Ask a question about the document")
        if user_pdf_prompt:
            st.chat_message("user").markdown(user_pdf_prompt)
            loading_pdf = st.empty()

            loading_pdf.chat_message("assistant", avatar=Symbol).markdown("Processing...")
            # Send the extracted PDF text and user question to the chatbot
            response_pdf = st.session_state.pdf_chat_session.send_message(user_pdf_prompt)
            loading_pdf.empty()
            writeAssistantResponse(response_pdf.text)
if st.session_state.current_page == "Research":
    st.image(Supreme_Court_Logo, width=100)
    st.title("Research Assistant")

    # Load structured instructions for the main chatbot
    with open('instructions4.txt', 'r') as file:
        STRUCTURED_INSTRUCTIONS_RESEARCH = file.read()

    # Main chatbot model
    research_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        system_instruction=STRUCTURED_INSTRUCTIONS_RESEARCH
    )

    # Maintain session for the main chatbot
    if "research_chat_session" not in st.session_state:
        st.session_state.research_chat_session = research_model.start_chat(history=[])

    for message in st.session_state.research_chat_session.history:
        text = message.parts[0].text.strip() if message.parts and message.parts[0].text else ""
        if text:
            with st.chat_message(translate_role_for_streamlit(message.role), avatar=Symbol if message.role == "model" else None):
                st.markdown(text)

    user_prompt = st.chat_input("What Do You Wanna Learn About?")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        loading = st.empty()

        loading.chat_message("assistant", avatar=Symbol).markdown("Processing...")
        response = st.session_state.research_chat_session.send_message(user_prompt)
        loading.empty()
        writeAssistantResponse(response.text)
    