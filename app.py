import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# Configuration: Hardcoded API Keys and URLs
DISCOVERY_API_KEY = '5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2'
WATSONX_API_KEY = 'zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz'
WATSONX_PROJECT_ID = '32a4b026-a46a-48df-aae3-31e16caabc3b'
DISCOVERY_SERVICE_URL = 'https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef'

# Watsonx Model Setup (Defaults)
DEFAULT_MODEL = "meta-llama/llama-3-1-70b-instruct"
url = "https://us-south.ml.cloud.ibm.com"
max_tokens = 100
min_tokens = 20
decoding = DecodingMethods.GREEDY
temperature = 0.7

# IBM Watson Discovery Setup
discovery_authenticator = IAMAuthenticator(DISCOVERY_API_KEY)
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=discovery_authenticator
)
discovery.set_service_url(DISCOVERY_SERVICE_URL)

# Define the model generator function
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
        credentials={"apikey": WATSONX_API_KEY, "url": url},
        project_id=WATSONX_PROJECT_ID
    )
    return model

# Streamlit UI setup
st.title("Watsonx AI and Discovery Integration")
st.write("This app allows you to ask questions, which will be answered by a combination of Watson Discovery and Watsonx model.")

# Sidebar Configuration
st.sidebar.title("Settings")
st.sidebar.markdown("### Model Settings")

# Dropdown for selecting the model
model_type = st.sidebar.selectbox(
    "Select Model",
    ["meta-llama/llama-3-1-70b-instruct", "meta-llama/llama-3-1-13b-instruct"],
    index=0
)

# Sliders for adjusting token limits
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=200, value=100)
min_tokens = st.sidebar.slider("Min Tokens", min_value=20, max_value=50, value=20)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Input for the question
question = st.text_input("Enter your question:")

# Session State for storing history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Button to clear the chat history
if st.sidebar.button("Clear Messages"):
    st.session_state['history'] = []

# Function to query Watson Discovery and Watsonx Model
def get_answer(question):
    try:
        # Query Watson Discovery
        response = discovery.query(
            project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
            collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
            passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
            natural_language_query=question
        ).get_result()

        # Process the Discovery response
        if 'results' in response and response['results']:
            passages = response['results'][0].get('document_passages', [])
            passages_texts = [p['passage_text'].replace('<em>', '').replace('</em>', '').replace('\n', '') for p in passages]
            context = '\n '.join(passages_texts)
        else:
            context = "No relevant information found in Discovery."

        # Prepare the prompt for Watsonx
        prompt = (
            "<s>[INST] <<SYS>> "
            "Please answer the following question in one sentence using this text. "
            "If the question is unanswerable, say 'unanswerable'. "
            "Question:" + question + 
            '<</SYS>>' + context + '[/INST]'
        )

        # Generate the answer using Watsonx
        model = get_model(model_type, max_tokens, min_tokens, decoding, temperature)
        generated_response = model.generate(prompt)
        response_text = generated_response['results'][0]['generated_text']

        # Save the question and answer to history
        st.session_state['history'].append({"question": question, "answer": response_text})
        return response_text

    except Exception as e:
        return f"Error fetching the answer: {e}"

# Display the question and generated answer
if st.button('Get Answer'):
    if question:
        answer = get_answer(question)
        st.subheader("Generated Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question!")

# Display chat history
st.subheader("Chat History")
for idx, entry in enumerate(st.session_state['history']):
    st.write(f"Q{idx + 1}: {entry['question']}")
    st.write(f"A{idx + 1}: {entry['answer']}")
    st.markdown("---")
