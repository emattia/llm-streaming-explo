import os
from datetime import datetime
import time
import gc 

import streamlit as st
import cohere
import pandas as pd

from helpers.cohere import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="Cohere Token Counter", page_icon="🤖")

# LHS navigation
with st.sidebar:
    cohere_api_key = os.environ.get("COHERE_API_KEY", None)
    if cohere_api_key is None:
        try:
            cohere_api_key = st.secrets["COHERE_API_KEY"]
        except KeyError:
            pass
    if cohere_api_key is None or cohere_api_key == "":
        openai_api_key = st.text_input("COHERE_API_KEY variable not detected. Set the environment variable or paste your key here:", key="api_key_cohere", type="password")
        os.environ["COHERE_API_KEY"] = cohere_api_key
        if cohere_api_key is not None and not cohere_api_key == "":
            st.balloons()
            time.sleep(2)
            st.rerun()
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/3_Cohere_Chat.py)"
    "[Cohere pricing data source](https://cohere.ai/pricing)"
    "[Cohere playground](https://dashboard.cohere.com/playground/chat)"

if cohere_api_key is None:
    st.stop()
else:
    client = cohere.Client(api_key=cohere_api_key)
    del cohere_api_key
    gc.collect()

# create a session state to log stuff
if "cohere_model" not in st.session_state or "cohere_observations" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://cohere.com/pricing to find the latest models and update COHERE_MODEL_DATA in openai-app.py""")
    
    st.session_state["cohere_model"] = DEFAULT_MODEL
    st.session_state["cohere_observations"] = []
    st.session_state.user_icon = open("icons/user.svg").read()
    st.session_state.content = ""
st.session_state.lm_icon = open("icons/cohere-square.svg").read()

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# system_msg, frequency_penalty, max_tokens, presence_penalty, stop_sequences, temperature, top_p = api_params_panel()
temperature = api_params_panel()

def clear():
    st.session_state.chat_history.clear()
    st.session_state.cohere_observations.clear()

reset_chat=st.button("Reset chat", on_click=clear)
submit=st.button("Submit")

with st.container():
    role = st.selectbox("Role", options=[USER_ROLE, LM_ROLE], label_visibility='collapsed')
    content = st.text_input(f"Enter {role} message here", "", key='content')

    if add_msg := st.button("Add message to chat history"):
        if content != "":
            st.session_state.chat_history.append({"role": role, "message": st.session_state.content})

    for message in st.session_state.chat_history:
        with st.chat_message(name=message["role"], avatar=get_icon(message["role"])):
            st.markdown(message["message"])

    prompt = st.text_input("Enter prompt here", "", key='prompt')

    if submit:
        with st.chat_message(LM_ROLE, avatar=get_icon(LM_ROLE)):
            api_params = dict(
                model=st.session_state["cohere_model"],
                message=prompt,
                chat_history=st.session_state.chat_history,
                temperature=temperature
            )
            with st.spinner("Generating response..."):
                response = client.chat(**api_params)
            st.markdown(response.text)
        estimated_input_tokens = num_tokens_from_prompt(prompt, st.session_state.chat_history)
        estimated_output_tokens = num_tokens_from_response(response.text)
        data_instance = TokenizersEstimateInstance( 
            time = datetime.now(),
            provider = PROVIDER,
            api_id = st.session_state["cohere_model"],
            message = prompt, 
            chat_history = st.session_state.chat_history,
            estimated_input_tokens = estimated_input_tokens,
            estimated_output_tokens = estimated_output_tokens,
            response = response.text,
            actual_prompt_tokens = response.meta['billed_units']['input_tokens'],
            actual_request_tokens = response.meta['billed_units']['output_tokens'],
        )
        st.session_state["cohere_observations"].append(data_instance)

if len(st.session_state["cohere_observations"]) > 0:

    st.divider()

    IDX = -1
    cohere_observation = st.session_state["cohere_observations"][IDX]

    # display the messages
    st.markdown("####  Most recent API request inputs")
    st.markdown("##### Chat history")
    st.json(cohere_observation.chat_history)
    st.markdown("##### Prompt")
    st.markdown(cohere_observation.message)

    st.markdown("##### Response")
    st.markdown(cohere_observation.response)

    # display the estimated and actual token counts
    all_data = [
        o.model_dump() for o in st.session_state["cohere_observations"]
    ]
    samples = {'estimated_input_tokens': [], 'estimated_output_tokens': [], 'actual_prompt_tokens': [], 'actual_request_tokens': []}
    for sample in all_data:
        print(sample)
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