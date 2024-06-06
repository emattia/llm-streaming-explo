import os
from datetime import datetime
import time
import gc

import streamlit as st
from openai import OpenAI
import pandas as pd

from helpers.openai import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="OpenAI Token Counter", page_icon="🤖")

# LHS navigation
with st.sidebar:
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except KeyError:
            pass
    if openai_api_key is None or openai_api_key == "":
        openai_api_key = st.text_input("OPENAI_API_KEY variable not detected. Set the environment variable or paste your key here:", key="api_key_openai", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if openai_api_key is not None and not openai_api_key == "":
            st.balloons()
            time.sleep(2)
            st.rerun()
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/2_OpenAI_Playground.py)"
    "[The actual OpenAI playground](https://platform.openai.com/playground)"

if openai_api_key is None:
    st.stop()
else:
    client = OpenAI(api_key=openai_api_key)
    del openai_api_key
    gc.collect()

# create a session state to log stuff
if "openai_model" not in st.session_state or "openai_observations" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://openai.com/pricing to find the latest models and update OPENAI_MODEL_DATA in openai-app.py""")
    
    st.session_state["openai_model"] = DEFAULT_MODEL
    st.session_state["openai_observations"] = []
    st.session_state.user_icon = open("icons/user.svg").read()
    st.session_state.content = ""
st.session_state.lm_icon = open("icons/chatgpt.svg").read()

# initialize chat history
if "openai_token_ctr_messages" not in st.session_state:
    st.session_state.openai_token_ctr_messages = []

system_msg, frequency_penalty, max_tokens, presence_penalty, stop_sequences, temperature, top_p = api_params_panel()

def clear():
    st.session_state.openai_token_ctr_messages.clear()
    st.session_state.openai_observations.clear()

reset_chat=st.button("Reset chat", on_click=clear)
submit=st.button("Submit")

with st.container():
    role = st.selectbox("Role", options=[USER_ROLE, LM_ROLE], label_visibility='collapsed')
    content = st.text_input(f"Enter {role} message here", "", key='content')

    if add_msg := st.button("Add message"):
        if content != "":
            st.session_state.openai_token_ctr_messages.append({"role": role, "content": st.session_state.content})

    for message in st.session_state.openai_token_ctr_messages:
        with st.chat_message(name=message["role"], avatar=get_icon(message["role"])):
            st.markdown(message["content"])

    if submit:
        message_payload = [{"role": SYSTEM_ROLE, "content": system_msg}] + st.session_state.openai_token_ctr_messages
        with st.chat_message(LM_ROLE, avatar=get_icon(LM_ROLE)):
            api_params = dict(
                model=st.session_state["openai_model"],
                messages=message_payload,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                temperature=temperature,
                top_p=top_p
            )
            with st.spinner("Generating response..."):
                response = client.chat.completions.create(**api_params)
            st.markdown(response.choices[0].message.content)
        message_payload = [{"role": SYSTEM_ROLE, "content": system_msg}] + st.session_state.openai_token_ctr_messages
        estimated_input_tokens = num_tokens_from_prompt(message_payload, st.session_state["openai_model"])
        estimated_output_tokens = num_tokens_from_string(response.choices[0].message.content, st.session_state["openai_model"])
        data_instance = TiktokenEstimateInstance( 
            time = datetime.now(),
            provider = PROVIDER,
            api_id = st.session_state["openai_model"],
            messages = message_payload,
            estimated_input_tokens = estimated_input_tokens,
            estimated_output_tokens = estimated_output_tokens,
            response = response.choices[0].message.content,
            actual_prompt_tokens = response.usage.prompt_tokens,
            actual_request_tokens = response.usage.completion_tokens,
            system_fingerprint = response.system_fingerprint
        )
        st.session_state["openai_observations"].append(data_instance)

if len(st.session_state["openai_observations"]) > 0:

    st.divider()

    IDX = -1
    openai_observation = st.session_state["openai_observations"][IDX]

    # display the openai_token_ctr_messages
    st.markdown("####  Most recent API request inputs")
    st.markdown("##### Messages")
    st.json(openai_observation.messages)

    st.markdown("##### Response")
    st.markdown(openai_observation.response)

    # display the estimated and actual token counts
    all_data = [
        o.model_dump() for o in st.session_state["openai_observations"]
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

    st.markdown("#### Estimated (tiktoken) and actual token counts")
    st.dataframe(df)

sidebar_footer()