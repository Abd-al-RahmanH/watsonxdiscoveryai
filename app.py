import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Streamlit UI setup
st.title("Watsonx AI and Discovery Integration")
st.write("This app allows you to ask questions, which will be answered by a combination of Watson Discovery and Watsonx model.")

# Sidebar Configuration
st.sidebar.title("Settings")

# Token inputs in the sidebar
discovery_token = st.sidebar.text_input("Discovery Token", type="password", value="", help="Enter your Watson Discovery token")
watsonx_api_key = st.sidebar.text_input("Watsonx API Key", type="password", value="", help="Enter your Watsonx API Key")
watsonx_project_id = st.sidebar.text_input("Watsonx Project ID", value="", help="Enter your Watsonx Project ID")
service_url = st.sidebar.text_input("Watson Discovery URL", value="https://api.us-south.discovery.watson.cloud.ibm.com")

# Model selection in the sidebar
models_list = [
    "meta-llama/llama-3-1-70b-instruct",
    "meta-llama/llama-2-70b-chat",
    "meta-llama/llama-2-13b-chat"
]
model_type = st.sidebar.selectbox("Select Model", models_list)

# Token settings
max_tokens = st.sidebar.slider("Max Tokens", 50, 200, 100)
min_tokens = st.sidebar.slider("Min Tokens", 10, 50, 20)
decoding = st.sidebar.selectbox("Decoding Method", list(DecodingMethods.__members__.values()), index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Clear message history button
if st.sidebar.button("Clear Messages"):
    st.session_state["history"] = []

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Discovery API Setup
authenticator = IAMAuthenticator(discovery_token)
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url(service_url)

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
        credentials={"apikey": watsonx_api_key, "url": "https://us-south.ml.cloud.ibm.com"},
        project_id=watsonx_project_id
    )
    return model

# Input for the question
question = st.text_input("Enter your question:")

# Fetch answer and update history
if st.button('Get Answer'):
    if question:
        try:
            # Query Watson Discovery
            response = discovery.query(
                project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
                collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
                passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
                natural_language_query=question
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
                "Do not include information that's not relevant to the question. "
                "Question:" + question + 
                '<</SYS>>' + context + '[/INST]'
            )

            # Generate the answer using Watsonx
            model = get_model(model_type, max_tokens, min_tokens, decoding, temperature)
            generated_response = model.generate(prompt)
            response_text = generated_response['results'][0]['generated_text']

            # Store in history
            st.session_state["history"].append({
                "question": question,
                "answer": response_text
            })

            # Display the generated response
            st.subheader("Generated Answer:")
            st.write(response_text)

        except Exception as e:
            st.error(f"Error fetching the answer: {str(e)}")
    else:
        st.error("Please enter a question!")

# Display history
if st.session_state["history"]:
    st.subheader("Answer History")
    for idx, qa in enumerate(st.session_state["history"], 1):
        st.write(f"**Q{idx}:** {qa['question']}")
        st.write(f"**A{idx}:** {qa['answer']}")
