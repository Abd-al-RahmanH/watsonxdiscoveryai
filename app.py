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
discovery.set_service_url('https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef')

# Watsonx Model Setup
url = "https://us-south.ml.cloud.ibm.com"
api_key = "zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz"
watsonx_project_id = "32a4b026-a46a-48df-aae3-31e16caabc3b"
model_type = "meta-llama/llama-3-1-70b-instruct"

# Streamlit UI setup
st.set_page_config(page_title="Watsonx AI and Discovery Integration", layout="wide")
st.title("Watsonx AI and Discovery Integration")

# Sidebar for selecting mode
mode = st.sidebar.radio("Select Mode", ["Watson Discovery", "LLM"])

# Sidebar for Model Parameters
st.sidebar.header("Watsonx Model Settings")
max_tokens = st.sidebar.slider("Max Output Tokens", 100, 4000, 600)
decoding = st.sidebar.radio("Decoding Method", ["greedy", "sample"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Clear Messages button in Sidebar
if st.sidebar.button("Clear Messages"):
    st.session_state.history = []

# Define the model generator function
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

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history with icons
for i, (user_msg, bot_msg, icon_user, icon_bot) in enumerate(st.session_state.history):
    st.write(f"{icon_user} User:", user_msg)
    st.write(f"{icon_bot} Assistant:", bot_msg)

# Input for the question
question = st.text_input("Ask your question here:")

if st.button('Get Answer'):
    if question:
        if mode == "LLM":
            # Display LLM invocation message
            st.info("Invoking LLM...")

            # Prepare prompt for Watsonx LLM
            prompt = (
                "<s>[INST] <<SYS>> "
                "Please answer the following question in one sentence using this text. "
                "If the question is unanswerable, say 'unanswerable'. "
                "Question:" + question + "<</SYS>>[/INST]"
            )

            # Generate answer from Watsonx LLM
            model = get_model(model_type, max_tokens, temperature)
            generated_response = model.generate(prompt)
            bot_response = generated_response['results'][0]['generated_text']

            # Add to history with icons
            st.session_state.history.append((question, bot_response, "🟥", "🟨"))

        elif mode == "Watson Discovery":
            # Query Watson Discovery
            response = discovery.query(
                project_id='016da9fc-26f5-464a-a0b8-c9b0b9da83c7',
                collection_ids=['1d91d603-cd71-5cf5-0000-019325bcd328'],
                passages={'enabled': True, 'max_per_document': 5, 'find_answers': True},
                natural_language_query=question
            ).get_result()

            # Process Discovery response
            try:
                passages = response['results'][0]['document_passages']
                passages = [p['passage_text'].replace('<em>', '').replace('</em>', '').replace('\n', '') for p in passages]
                bot_response = '\n'.join(passages)
            except IndexError:
                bot_response = "No relevant document found in Watson Discovery."

            # Add to history with icons
            st.session_state.history.append((question, bot_response, "🟥", "🟨"))

        # Display the generated response
        st.write("Generated Answer:")
        st.write(bot_response)
    else:
        st.error("Please enter a question!")
