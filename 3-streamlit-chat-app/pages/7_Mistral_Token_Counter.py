import os
from datetime import datetime
import time
import gc

import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralException
import pandas as pd

from helpers.mistral import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="Mistral Token Counter", page_icon="🤖")

# LHS navigation
with st.sidebar:
    mistral_api_key = os.environ.get("MISTRAL_API_KEY", None)
    if mistral_api_key is None:
        try: 
            mistral_api_key = st.secrets["MISTRAL_API_KEY"]
        except KeyError:
            pass
    if mistral_api_key is None or mistral_api_key == "":        
        mistral_api_key = st.text_input("MISTRAL_API_KEY variable not detected. Get it [here](https://console.mistral.ai/user/api-keys/). Set the environment variable or paste your key here:", key="api_key_mistral", type="password")
        os.environ["MISTRAL_API_KEY"] = mistral_api_key
        if mistral_api_key is not None and not mistral_api_key == "":
            st.balloons()
            time.sleep(2)
            st.rerun()

    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/6_Mistral_Token_Counter.py)"
    "[Mistral pricing data source](https://docs.mistral.ai/platform/pricing/)"

if mistral_api_key is None or mistral_api_key == "":
    st.stop()
else:
    client = MistralClient(api_key=mistral_api_key)
    del mistral_api_key
    gc.collect()

# create a session state to log stuff
if "mistral_model" not in st.session_state or "mistral_observations" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://docs.mistral.ai/platform/pricing/ to find the latest models and update mistral_MODEL_DATA in mistral-app.py""")
    
    st.session_state["mistral_model"] = DEFAULT_MODEL
    st.session_state["mistral_observations"] = []
    st.session_state.user_icon = open("icons/user.svg").read()
    st.session_state.content = ""
st.session_state.lm_icon = open("icons/mistral.svg").read()

# initialize chat history
if "mistral_messages" not in st.session_state:
    st.session_state.mistral_messages = []

system_msg = api_params_panel()

def clear():
    st.session_state.mistral_messages.clear()
    st.session_state.mistral_observations.clear()

reset_chat=st.button("Reset chat", on_click=clear)

with st.container():

    role = st.selectbox("Role", options=[USER_ROLE, LM_ROLE], label_visibility='collapsed')
    content = st.text_input(f"Enter {role} message here", "", key='content')

    if add_msg := st.button("Add message"):
        if content != "":
            st.session_state.mistral_messages.append(ChatMessage(role=role, content=st.session_state.content))
    
    for message in st.session_state.mistral_messages:
        with st.chat_message(name=message.role, avatar=get_icon(message.role)):
            st.markdown(message.content)

    submit=st.button("Submit")

    if submit:
        message_payload = [ChatMessage(role=SYSTEM_ROLE, content=system_msg)] + st.session_state.mistral_messages
        print(message_payload)
        with st.chat_message(LM_ROLE, avatar=get_icon(LM_ROLE)):
            api_params = dict(
                model=st.session_state["mistral_model"],
                messages=message_payload
            )
            with st.spinner("Generating response..."):
                response = client.chat(**api_params)
            st.markdown(response.choices[0].message.content)
        message_payload = [ChatMessage(role=SYSTEM_ROLE, content=system_msg)] + st.session_state.mistral_messages
        estimated_input_tokens = num_tokens_from_prompt(message_payload, st.session_state["mistral_model"])
        estimated_output_tokens = num_tokens_from_string(response.choices[0].message.content, st.session_state["mistral_model"])
        data_instance = TokenizersEstimateInstance( 
            time = datetime.now(),
            provider = PROVIDER,
            api_id = st.session_state["mistral_model"],
            messages = [m.model_dump() for m in message_payload],
            estimated_input_tokens = estimated_input_tokens,
            estimated_output_tokens = estimated_output_tokens,
            response = response.choices[0].message.content,
            actual_prompt_tokens = response.usage.prompt_tokens,
            actual_request_tokens = response.usage.completion_tokens
        )
        st.session_state["mistral_observations"].append(data_instance)

if len(st.session_state["mistral_observations"]) > 0:

    st.divider()

    IDX = -1
    mistral_observation = st.session_state["mistral_observations"][IDX]

    # display the mistral_messages
    st.markdown("####  Most recent API request inputs")
    st.markdown("##### Messages")
    st.json(mistral_observation.messages)

    st.markdown("##### Response")
    st.markdown(mistral_observation.response)

    # display the estimated and actual token counts
    all_data = [
        o.model_dump() for o in st.session_state["mistral_observations"]
    ]
    samples = {'estimated_input_tokens': [], 'estimated_output_tokens': [], 'actual_prompt_tokens': [], 'actual_request_tokens': []}
    for sample in all_data:
        samples['estimated_input_tokens'].append(sample['estimated_input_tokens'])
        samples['estimated_output_tokens'].append(sample['estimated_output_tokens'])
        samples['actual_prompt_tokens'].append(sample['actual_prompt_tokens'])
        samples['actual_request_tokens'].append(sample['actual_request_tokens'])

    df = pd.DataFrame(samples)
    df['underestimate_input'] =  df['actual_prompt_tokens'] - df['estimated_input_tokens']
    df['underestimate_output'] =  df['actual_request_tokens'] - df['estimated_output_tokens']

    st.markdown("#### Estimated and actual token counts")
    st.dataframe(df)

sidebar_footer()