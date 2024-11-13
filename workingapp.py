import streamlit as st
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# IBM Watson Discovery and Watsonx Credentials (replace with your keys)
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
max_tokens = 100
min_tokens = 50
decoding = DecodingMethods.GREEDY
temperature = 0.7

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
        credentials={"apikey": api_key, "url": url},
        project_id=watsonx_project_id
    )
    return model

# Streamlit UI setup
st.title("Watsonx AI and Discovery Integration")
st.write("This app allows you to ask questions, which will be answered by a combination of Watson Discovery and Watsonx model.")

# Input for the question
question = st.text_input("Enter your question:")

if st.button('Get Answer'):
    if question:
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
            "If you responded to the question, don't say 'unanswerable'. "
            "Do not include information that's not relevant to the question. "
            "Do not answer other questions. "
            "Make sure the language used is English.'"
            "Do not use repetitions' "
            "Question:" + question + 
            '<</SYS>>' + context + '[/INST]'
        )

        # Generate the answer using Watsonx
        model = get_model(model_type, max_tokens, min_tokens, decoding, temperature)
        generated_response = model.generate(prompt)
        response_text = generated_response['results'][0]['generated_text']

        # Display the generated response
        st.subheader("Generated Answer:")
        st.write(response_text)
    else:
        st.error("Please enter a question!")

