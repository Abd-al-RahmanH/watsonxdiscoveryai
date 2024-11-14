import streamlit as st
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model

# Hardcoded credentials
IBM_DISC_API_KEY = '5sSmoI6y0ZHP7D3a6Iu80neypsbK3tsUZR_VdRAb7ed2'
IBM_DISC_PROJ_ID = '016da9fc-26f5-464a-a0b8-c9b0b9da83c7'
IBM_DISC_SERVICE_URL = 'https://api.us-south.discovery.watson.cloud.ibm.com/instances/62dc0387-6c6f-4128-b479-00cf5dea09ef'
IBM_DISC_COLL_ID = '1d91d603-cd71-5cf5-0000-019325bcd328'

IBM_WX_API_KEY = 'zf-5qgRvW-_RMBGb0bQw5JPPGGj5wdYpLVypdjQxBGJz'
IBM_WX_PROJ_ID = '32a4b026-a46a-48df-aae3-31e16caabc3b'
IBM_WX_SERVICE_URL = 'https://us-south.ml.cloud.ibm.com'

# Authentication with services
authenticator = IAMAuthenticator(IBM_DISC_API_KEY)
discovery = DiscoveryV2(version='2020-08-30', authenticator=authenticator)
discovery.set_service_url(IBM_DISC_SERVICE_URL)
discovery.set_disable_ssl_verification(True)

ai_credentials = Credentials(
    url=IBM_WX_SERVICE_URL,
    api_key=IBM_WX_API_KEY
)

# Define the prompt template
PROMPT_TEMPLATE = '''
CONTEXT:
%s

QUESTION:
%s

INSTRUCTIONS:
Answer the user's QUESTION using the CONTEXT text above.
Keep your answer grounded in the facts of the CONTEXT.
If the CONTEXT doesn't contain the facts to answer the QUESTION return "I don't know".

ANSWER:
'''

# Streamlit app UI
st.title("Watsonx AI & Discovery - RAG Query Answering")
st.write("Enter a question, and the system will try to answer using IBM Watson Discovery and Watsonx AI.")

# Input: Question
QUERY = st.text_input("Enter your question:")

if QUERY:
    # Get passages from Watson Discovery
    passage_list = discovery.query(
        project_id=IBM_DISC_PROJ_ID,
        natural_language_query=QUERY,
        count=5,
        collection_ids=[IBM_DISC_COLL_ID],
        similar={"fields": ["text"]},
        passages={
            "enabled": True,
            "per_document": True,
            "find_answers": True,
            "max_answers_per_passage": 1,
            "characters": 250
        }
    ).get_result()['results']

    # Wrap into singular string
    passage_text = '\n\n'.join(list(map(
        lambda x: '\n\n'.join(map(lambda p: p['passage_text'], x['document_passages'])),
        passage_list
    )))

    # Define model parameters
    model = Model(
        model_id="meta-llama/llama-3-1-70b-instruct",
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "temperature": 0,
            "min_new_tokens": 35,
            "repetition_penalty": 1.1,
            "stop_sequences": ["\n\n"]
        },
        project_id=IBM_WX_PROJ_ID,
        credentials=ai_credentials
    )

    # Combine passage text and query into final prompt
    final_prompt = PROMPT_TEMPLATE % (passage_text, QUERY)

    # Get response from the model
    output = model.generate_text(prompt=final_prompt, guardrails=False)

    # Display the result
    st.write("### Answer:")
    st.write(output)
