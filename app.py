import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

# Configuration
DISCOVERY_API_KEY = '5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2'
WATSONX_API_KEY = 'zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz'
WATSONX_PROJECT_ID = '32a4b026-a46a-48df-aae3-31e16caabc3b'
DISCOVERY_SERVICE_URL = 'https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef'
url = "https://us-south.ml.cloud.ibm.com"

# Streamlit Config
st.set_page_config(page_title="Watsonx Advanced UI", layout="wide")

# Sidebar - Model Selection
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Select Model", 
    ["meta-llama/llama-3-1-70b-instruct", "gpt-j-6b-instruct", "gpt-neo-2-7b"]
)
max_tokens = st.sidebar.slider("Max Tokens", 50, 4000, 1000)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Watson Discovery Setup
authenticator = IAMAuthenticator(DISCOVERY_API_KEY)
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url(DISCOVERY_SERVICE_URL)

# Watsonx Model Setup
def get_model():
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: 20,
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.TEMPERATURE: temperature,
    }
    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={"apikey": WATSONX_API_KEY, "url": url},
        project_id=WATSONX_PROJECT_ID
    )
    return model

# Chat UI Container
st.markdown("""
    <style>
        .main-chat-container { max-width: 800px; margin: 0 auto; }
        .chat-bubble { padding: 12px; border-radius: 12px; margin-bottom: 10px; font-size: 16px; }
        .user-message { background-color: #0078D7; color: white; text-align: right; }
        .ai-message { background-color: #444; color: white; }
        .prompt-buttons { display: flex; gap: 10px; }
        .prompt-button { padding: 10px; border-radius: 10px; background-color: #0078D7; color: white; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

# Suggested Prompts
st.markdown("<h1 style='text-align: center;'>Watsonx Advanced UI</h1>", unsafe_allow_html=True)
st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)
suggested_prompts = ["Help me study", "Tell me a fun fact", "Overcome procrastination", "Give me ideas"]
st.markdown("<div class='prompt-buttons'>", unsafe_allow_html=True)
for prompt in suggested_prompts:
    if st.button(prompt):
        question = prompt
st.markdown("</div>", unsafe_allow_html=True)

# Text Input
question = st.text_input("Ask your question here...")

# Get Answer Function
def get_answer(question):
    try:
        # Watson Discovery Query
        response = discovery.query(
            project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
            collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
            passages={'enabled': True},
            natural_language_query=question
        ).get_result()
        
        # Extract Context
        passages = response['results'][0].get('document_passages', [])
        context = '\n'.join([p['passage_text'] for p in passages]) or "No relevant information found."
        
        # Watsonx Prompt
        prompt = f"<s>[INST] <<SYS>> Please answer the question in a concise manner: {question} <<SYS>> {context} [/INST]"
        model = get_model()
        generated_response = model.generate(prompt)
        return generated_response['results'][0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Display Chat
if question:
    answer = get_answer(question)
    st.markdown(f"<div class='chat-bubble user-message'>{question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble ai-message'>{answer}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
