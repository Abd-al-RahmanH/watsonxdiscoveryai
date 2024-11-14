import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# IBM Watson Discovery and Watsonx Credentials (hardcoded for development)
authenticator = IAMAuthenticator('5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2')
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url('https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef')
url = "https://us-south.ml.cloud.ibm.com"
api_key = "zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz"
watsonx_project_id = "32a4b026-a46a-48df-aae3-31e16caabc3b"

# Sidebar Setup for Model Selection and Token Count Display
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    options=["meta-llama/llama-3-1-70b-instruct", "meta-gpt/gpt-3.5-turbo", "meta-bert/bert-large"],
    index=0
)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=230, step=10)
min_tokens = st.sidebar.slider("Min Tokens", min_value=10, max_value=100, value=50, step=5)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
decoding = DecodingMethods.GREEDY

# Clear Messages Button in Sidebar
if st.sidebar.button("Clear Messages"):
    st.session_state.history = []

# Initialize message history
if "history" not in st.session_state:
    st.session_state.history = []
if "input_question" not in st.session_state:
    st.session_state.input_question = ""

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

# Main App Layout
st.title("Watsonx AI and Discovery Integration")
st.write("This app combines IBM Watson Discovery with Watsonx AI for question answering.")

# Display chat history
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Answer:** {entry['response']}")

# Input for the question with a temporary key
question_temp = st.text_input("Enter your question:", key="input_question_temp")

# Submit button for the question
if st.button("Submit Question"):
    # Save question to session state
    st.session_state.input_question = question_temp

    # Query Watson Discovery
    response = discovery.query(
        project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
        collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
        passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
        natural_language_query=st.session_state.input_question
    ).get_result()

    # Process the Discovery response
    passages = response['results'][0]['document_passages']
    passages = [p['passage_text'].replace('<em>', '').replace('</em>', '').replace('\n', '') for p in passages]
    context = '\n '.join(passages)

    # Prepare the prompt for Watsonx
    prompt = (
        "<s>[INST] <<SYS>> "
        "Please answer the following question in one sentence using this text. "
        "If the question is unanswerable, say 'unanswerable'. "
        "If you responded to the question, don't say 'unanswerable'. "
        "Do not include information that's not relevant to the question. "
        "Do not answer other questions. "
        "Make sure the language used is English.'"
        "Do not use repetitions' "
        "Question:" + st.session_state.input_question + 
        '<</SYS>>' + context + '[/INST]'
    )

    # Generate the answer using Watsonx
    model = get_model(model_type, max_tokens, min_tokens, decoding, temperature)
    generated_response = model.generate(prompt)
    response_text = generated_response['results'][0]['generated_text']

    # Append to history
    st.session_state.history.append({"question": st.session_state.input_question, "response": response_text})
    
    # Clear the input question
    st.session_state.input_question = ""
    st.session_state.input_question_temp = ""  # Clear the temporary input

# This version should avoid the error and clear the question input field after submission.
