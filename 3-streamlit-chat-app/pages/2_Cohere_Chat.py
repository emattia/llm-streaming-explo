import os
from datetime import datetime
import gc
import time

import streamlit as st
import cohere
import pandas as pd

from helpers.cohere import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="OpenAI Chat", page_icon="🤖")

# LHS navigation
with st.sidebar:
    cohere_api_key = get_cohere_key()
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/2_Cohere_Chat.py)"
    "[Cohere pricing data source](https://cohere.ai/pricing)"

# create a session state to log stuff
if "cohere_model" not in st.session_state or "cohere_prompt_tokens" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://cohere.com/pricing to find the latest models and update COHERE_MODEL_DATA in helpers/cohere.py""")
    st.session_state["cohere_model"] = DEFAULT_MODEL
    st.session_state["cohere_prompt_tokens"] = 0
    st.session_state["cohere_response_tokens"] = 0
    st.session_state["prompt_cost"] = MODEL_DATA[DEFAULT_MODEL]["input"]
    st.session_state["response_cost"] = MODEL_DATA[DEFAULT_MODEL]["output"]
    st.session_state.chat_history = [{"role": LM_ROLE, "message": BOT_OPENING_LINE}]
    st.session_state.user_icon = open("icons/user.svg").read()
st.session_state.lm_icon = open("icons/cohere-square.svg").read()

if cohere_api_key is None:
    st.stop()
else:
    client = cohere.Client(api_key=cohere_api_key)
    del cohere_api_key
    gc.collect()

temperature = api_params_panel()

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": LM_ROLE, "message": BOT_OPENING_LINE}]

# write out the messages
for message in st.session_state.chat_history:
    with st.chat_message(name=message["role"], avatar=get_icon(message["role"])):
        st.markdown(message["message"])

if client:

    if prompt := st.chat_input(META_PROMPT):

        # get user prompt
        with st.chat_message(name=USER_ROLE, avatar=get_icon(USER_ROLE)):
            st.markdown(prompt)

        # chat iteration
        with st.chat_message(name=LM_ROLE, avatar=get_icon(LM_ROLE)):
            message_placeholder = st.empty()
            full_response = ""
            api_params = dict(
                model=st.session_state["cohere_model"], 
                chat_history=st.session_state.chat_history,
                message=prompt, 
                # stream=True,
                temperature=temperature
            )
            num_chunks_streamed = 0; t0 = datetime.now()

            # for response in client.chat(**api_params):
            for response in client.chat_stream(**api_params):
                if response.event_type == 'stream-start':
                    continue
                elif response.event_type == 'stream-end':
                    break
                elif response.event_type == 'text-generation':
                    full_response += response.text or ""
                    message_placeholder.markdown(full_response + '|')
                    num_chunks_streamed += 1
            tf = datetime.now()
            message_placeholder.markdown(full_response)

        # add prompt to chat history
        st.session_state.chat_history.append({"role": USER_ROLE, "message": prompt})

        # add response it to chat history
        st.session_state.chat_history.append({"role": LM_ROLE, "message": full_response})

         # track token count - BUG not tracking tokens correctly
        n_prompt_tokens = num_tokens_from_prompt(prompt, st.session_state.chat_history)
        st.session_state["cohere_prompt_tokens"] += n_prompt_tokens
        
        # track token count
        n_response_tokens = num_tokens_from_response(full_response)
        st.session_state["cohere_response_tokens"] += n_response_tokens

        data_instance = DataInstance(
            time = t0,
            provider = PROVIDER,
            api_id = st.session_state["cohere_model"],
            prompt = prompt,
            response = full_response,
            estimated_input_tokens = n_prompt_tokens,
            estimated_output_tokens = n_response_tokens,
            estimated_input_cost = st.session_state['prompt_cost'] * n_prompt_tokens / COHERE_TOKENS_PRICING_UNIT,
            estimated_output_cost = st.session_state['response_cost'] * n_response_tokens / COHERE_TOKENS_PRICING_UNIT,
            generation_time = (tf - t0).total_seconds(),
            num_chunks_streamed = num_chunks_streamed,
            system_fingerprint = None
        )
        update_data_store(data_instance)

with st.sidebar:
    show_session_cost_stats(tokens_pricing_unit=COHERE_TOKENS_PRICING_UNIT, prompt_tokens=st.session_state["cohere_prompt_tokens"], response_tokens=st.session_state["cohere_response_tokens"])
sidebar_footer()