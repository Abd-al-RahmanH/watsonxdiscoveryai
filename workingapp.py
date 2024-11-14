import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# API Keys and Configuration (Hidden from UI)
DISCOVERY_API_KEY = '5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2'
DISCOVERY_SERVICE_URL = 'https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef'
WATSONX_API_KEY = "zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = "32a4b026-a46a-48df-aae3-31e16caabc3b"
WATSONX_MODEL_TYPE = "meta-llama/llama-3-1-70b-instruct"
MAX_TOKENS = 100
MIN_TOKENS = 50
DECODING = DecodingMethods.GREEDY
TEMPERATURE = 0.7

# Initialize Watson Discovery
authenticator = IAMAuthenticator(DISCOVERY_API_KEY)
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)
discovery.set_service_url(DISCOVERY_SERVICE_URL)

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
        credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
        project_id=WATSONX_PROJECT_ID
    )
    return model

# Streamlit UI setup
st.title("Watsonx AI and Discovery Integration")
st.write("This app allows you to ask questions, which will be answered by a combination of Watson Discovery and Watsonx model.")

# Input for the question
question = st.text_input("Enter your question:")

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
            model = get_model(WATSONX_MODEL_TYPE, MAX_TOKENS, MIN_TOKENS, DECODING, TEMPERATURE)
            generated_response = model.generate(prompt)
            response_text = generated_response['results'][0]['generated_text']

            # Display the generated response
            st.subheader("Generated Answer:")
            st.write(response_text)

        except Exception as e:
            st.error(f"Error fetching the answer: {str(e)}")
    else:
        st.error("Please enter a question!")
