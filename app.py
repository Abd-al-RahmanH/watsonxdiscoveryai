import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# Configuration: API Keys and URLs
DISCOVERY_API_KEY = '5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2'
WATSONX_API_KEY = 'zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz'
WATSONX_PROJECT_ID = '32a4b026-a46a-48df-aae3-31e16caabc3b'
DISCOVERY_SERVICE_URL = 'https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef'

# Watsonx Model Setup
url = "https://us-south.ml.cloud.ibm.com"
DEFAULT_MODEL = "meta-llama/llama-3-1-70b-instruct"
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
def get_model():
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
    }
    model = Model(
        model_id=DEFAULT_MODEL,
        params=generate_params,
        credentials={"apikey": WATSONX_API_KEY, "url": url},
        project_id=WATSONX_PROJECT_ID
    )
    return model

# Function to query Watson Discovery and Watsonx Model
def get_answer(question):
    try:
        response = discovery.query(
            project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
            collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
            passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
            natural_language_query=question
        ).get_result()

        passages = response['results'][0].get('document_passages', [])
        context = '\n'.join([p['passage_text'] for p in passages]) or "No relevant information found."

        prompt = (
            "<s>[INST] <<SYS>> "
            "Answer the question briefly. If you can't, say 'unanswerable'. "
            "Question: " + question + '<</SYS>>' + context + '[/INST]'
        )

        model = get_model()
        generated_response = model.generate(prompt)
        return generated_response['results'][0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI setup
st.set_page_config(page_title="Advanced AI Chat", layout="wide")
st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.5;
    }
    .user-question {
        background-color: #007bff;
        color: white;
        align-self: flex-end;
    }
    .ai-response {
        background-color: #f1f1f1;
        color: black;
    }
    .input-container {
        display: flex;
        justify-content: center;
        position: fixed;
        bottom: 10px;
        width: 100%;
    }
    .input-box {
        width: 70%;
        padding: 10px;
        border-radius: 20px;
        border: 1px solid #ddd;
        outline: none;
        font-size: 16px;
    }
    .send-button {
        padding: 10px 20px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        font-size: 16px;
        margin-left: 10px;
    }
    .send-button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state['history'] = []

question = st.text_input("", placeholder="Type your question here...", key="input")

if st.button("Send"):
    if question.strip():
        response = get_answer(question.strip())
        st.session_state['history'].append({"question": question, "answer": response})

for entry in st.session_state['history']:
    st.markdown(f'<div class="chat-bubble user-question">{entry["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble ai-response">{entry["answer"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
