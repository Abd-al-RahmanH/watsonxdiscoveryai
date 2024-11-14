import streamlit as st
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# IBM Watson Discovery Credentials
authenticator = IAMAuthenticator('5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2')
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url('https://api.us-south.discovery.watson.cloud.ibm.com/instances/your_instance')

# Watsonx Model Setup
url = "https://us-south.ml.cloud.ibm.com"
api_key = "zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz"
watsonx_project_id = "32a4b026-a46a-48df-aae3-31e16caabc3b"
model_type = "meta-llama/llama-3-1-70b-instruct"

# Streamlit UI setup
st.set_page_config(page_title="Watsonx AI and Discovery Integration", layout="wide")
st.title("Watsonx AI and Discovery Integration")

# Sidebar for selecting mode and uploading files
with st.sidebar:
    st.header("Document Uploader and Mode Selection")
    mode = st.radio("Select Mode", ["Watson Discovery", "LLM"], index=0)

    # File upload for document retrieval in LLM mode
    uploaded_file = st.file_uploader("Upload file for RAG", accept_multiple_files=False, type=["pdf", "docx", "txt", "pptx", "csv", "json", "xml", "yaml", "html"])
    
    # Sidebar for Model Parameters in LLM mode
    if mode == "LLM":
        st.header("Watsonx Model Settings")
        max_tokens = st.slider("Max Output Tokens", 100, 4000, 600)
        decoding = st.radio("Decoding Method", ["greedy", "sample"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

        # Watsonx model generator
        def get_model(model_type, max_tokens, temperature):
            generate_params = {
                GenParams.MAX_NEW_TOKENS: max_tokens,
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

# Main Chat Section
st.header("Chat with Watsonx AI or Discovery")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages
for message in st.session_state.history:
    if message["role"] == "user":
        st.chat_message(message["role"], avatar="🟦").markdown(message["content"])
    else:
        st.chat_message(message["role"], avatar="🟨").markdown(message["content"])

# Text input for questions
prompt = st.chat_input("Ask your question here", disabled=False if mode == "LLM" or mode == "Watson Discovery" else True)

# Button for query submission and generating responses
if prompt:
    st.chat_message("user", avatar="🟦").markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    if mode == "LLM":
        model = get_model(model_type, max_tokens, temperature)
        prompt_text = f"<s>[INST] <<SYS>> Please answer the question: {prompt}<</SYS>>[/INST]"
        response = model.generate(prompt_text)
        response_text = response['results'][0]['generated_text']

    elif mode == "Watson Discovery":
        query_response = discovery.query(
            project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
            natural_language_query=prompt,
            count=1
        ).get_result()
        if query_response['results']:
            response_text = query_response['results'][0]['text']
        else:
            response_text = "No relevant documents found."

    st.session_state.history.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant", avatar="🟨").markdown(response_text)

# Button to clear chat history
if st.sidebar.button("Clear Messages"):
    st.session_state.history = []
