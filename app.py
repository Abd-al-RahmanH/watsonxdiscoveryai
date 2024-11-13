import os
import streamlit as st
import tempfile
import json
import pandas as pd
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Watson Discovery and Watsonx API Setup
authenticator = IAMAuthenticator('5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2')
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url('https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef')

# Watsonx Model API Setup
url = "https://us-south.ml.cloud.ibm.com"
api_key = "zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz"
watsonx_project_id = "32a4b026-a46a-48df-aae3-31e16caabc3b"
model_type = "meta-llama/llama-3-1-70b-instruct"
max_tokens = 100
min_tokens = 50
decoding = DecodingMethods.GREEDY
temperature = 0.7

# Function to get Watsonx model
def get_model(model_type, max_tokens, min_tokens, decoding, temperature):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
    }
    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={"apikey": api_key, "url": url},
        project_id=watsonx_project_id
    )
    return model

# Streamlit UI Setup
st.title("Watsonx AI and Discovery Integration")
st.write("Ask questions and get responses using Watson Discovery and Watsonx AI models.")

# Sidebar Settings
with st.sidebar:
    st.title("Watsonx Settings")
    model_name = st.selectbox("Choose Model", ["meta-llama/llama-3-1-70b-instruct", "codellama/codellama-34b-instruct-hf", "ibm/granite-20b-multilingual"])
    max_new_tokens = st.slider("Max output tokens", min_value=100, max_value=4000, value=600, step=100)
    decoding_method = st.radio("Decoding Method", [DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# File upload section
uploaded_file = st.file_uploader("Upload a file for RAG", type=["pdf", "docx", "txt", "pptx", "csv", "json", "xml", "yaml", "html"])

# Load Watson Discovery Documents
if uploaded_file:
    st.write(f"File uploaded: {uploaded_file.name}")
    
    # You can implement your custom file loaders or use Langchain to load documents here
    # For simplicity, let's assume we're querying Watson Discovery for the file content
    
    question = st.text_input("Enter your question:")
    
    if question:
        # Query Watson Discovery
        response = discovery.query(
            project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
            collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
            passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
            natural_language_query=question
        ).get_result()
        
        # Extract document names from Watson Discovery response
        doc_names = [doc['id'] for doc in response['results']]
        st.write("Documents Retrieved from Watson Discovery:")
        st.write(doc_names)

        # Process the Discovery results
        passages = response['results'][0]['document_passages']
        passages = [p['passage_text'].replace('<em>', '').replace('</em>', '').replace('\n', '') for p in passages]
        context = '\n '.join(passages)
        
        # Generate the response using Watsonx AI model
        prompt = (
            "<s>[INST] <<SYS>> "
            "Please answer the following question in one sentence using this text. "
            "If the question is unanswerable, say 'unanswerable'. "
            "If you responded to the question, don't say 'unanswerable'. "
            "Do not include information that's not relevant to the question. "
            "Do not answer other questions. "
            "Make sure the language used is English.'"
            "Do not use repetitions' "
            "Question:" + question + 
            '<</SYS>>' + context + '[/INST]'
        )
        
        # Get the model and generate the answer
        model = get_model(model_name, max_new_tokens, min_tokens, decoding_method, temperature)
        generated_response = model.generate(prompt)
        response_text = generated_response['results'][0]['generated_text']
        
        # Display the answer
        st.subheader("Generated Answer:")
        st.write(response_text)

# Conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat loop
prompt = st.chat_input("Ask your question:")

if prompt:
    st.chat_message("user").markdown(prompt)
    model = get_model(model_name, max_new_tokens, min_tokens, decoding_method, temperature)
    generated_response = model.generate(prompt)
    response_text = generated_response['results'][0]['generated_text']
    
    # Display the response and update conversation history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({'role': 'assistant', 'content': response_text})
