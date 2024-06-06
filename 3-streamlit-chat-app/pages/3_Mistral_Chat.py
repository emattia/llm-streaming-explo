import os
from datetime import datetime
import gc
import time

import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralException
import pandas as pd

from helpers.mistral import *
from helpers.common import *


### START PAGE CONTENT ### 
st.set_page_config(page_title="Mistral Chat", page_icon="🤖")

with st.sidebar:
    mistral_api_key = get_mistral_key()
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/3_Mistral_Chat.py)"
    "[Mistral pricing data source](https://docs.mistral.ai/platform/pricing/)"

if mistral_api_key is None or mistral_api_key == "":
    st.stop()
else:
    client = MistralClient(api_key=mistral_api_key)
    del mistral_api_key
    gc.collect()

# create a session state to log stuff
if "mistral_model" not in st.session_state or "mistral_prompt_tokens" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://docs.mistral.ai/platform/pricing/ to find the latest models and update MODEL_DATA in helpers/mistral.py""")
    st.session_state["mistral_model"] = DEFAULT_MODEL
    st.session_state["mistral_prompt_tokens"] = 0
    st.session_state["mistral_response_tokens"] = 0
    st.session_state["prompt_cost"] = MODEL_DATA[DEFAULT_MODEL]["input"]
    st.session_state["response_cost"] = MODEL_DATA[DEFAULT_MODEL]["output"]
    st.session_state.mistral_messages = [ChatMessage(role=LM_ROLE, content=BOT_OPENING_LINE)]
    st.session_state.user_icon = open("icons/user.svg").read()
st.session_state.lm_icon = open("icons/mistral.svg").read()

# set up side bar
system_msg = api_params_panel()

# initialize chat history
if "mistral_messages" not in st.session_state:
    st.session_state.mistral_messages = [
        ChatMessage(role=LM_ROLE, content=BOT_OPENING_LINE)
    ]

# write out the mistral_messages
for message in st.session_state.mistral_messages:
    if isinstance(message, dict):
        message = ChatMessage(**message)
    with st.chat_message(name=message.role, avatar=get_icon(message.role)):
        st.markdown(message.content)

if client:

    if prompt := st.chat_input(META_PROMPT):

        # get user prompt
        with st.chat_message(name=USER_ROLE, avatar=get_icon(USER_ROLE)):
            st.markdown(prompt)

        # add prompt to chat history
        st.session_state.mistral_messages.append(ChatMessage(role=USER_ROLE, content=prompt))
        message_payload = [ChatMessage(role=SYSTEM_ROLE, content=system_msg)] + st.session_state.mistral_messages

        # track token count
        # n_prompt_tokens = num_tokens_from_prompt(message_payload, st.session_state["mistral_model"])
        # st.session_state["mistral_prompt_tokens"] += n_prompt_tokens
        
        # chat iteration
        with st.chat_message(name=LM_ROLE, avatar=get_icon(LM_ROLE)):
            message_placeholder = st.empty()
            full_response = ""
            api_params = dict( 
                model=st.session_state["mistral_model"], 
                messages=message_payload, 
            )
            num_chunks_streamed = 0; t0 = datetime.now()
            try:
                for response in client.chat_stream(**api_params):
                    full_response += response.choices[0].delta.content or ""
                    message_placeholder.markdown(full_response + '|')
                    num_chunks_streamed += 1
            except MistralException as e:
                st.error(e)
                st.markdown("#### Please try to reset your API token.")
                st.stop()
            tf = datetime.now()
            message_placeholder.markdown(full_response)

        # add response it to chat history
        st.session_state.mistral_messages.append(ChatMessage(role=LM_ROLE, content=full_response))

        # track token count
        # n_response_tokens = num_tokens_from_string(full_response, st.session_state["mistral_model"])
        # st.session_state["mistral_response_tokens"] += n_response_tokens

        # data_instance = DataInstance(
        #     time = t0,
        #     provider = PROVIDER,
        #     api_id = st.session_state["mistral_model"],
        #     prompt = prompt,
        #     response = full_response,
        #     estimated_input_tokens = n_prompt_tokens,
        #     estimated_output_tokens = n_response_tokens,
        #     estimated_input_cost = st.session_state['prompt_cost'] * n_prompt_tokens / MISTRAL_TOKENS_PRICING_UNIT,
        #     estimated_output_cost = st.session_state['response_cost'] * n_response_tokens / MISTRAL_TOKENS_PRICING_UNIT,
        #     generation_time = (tf - t0).total_seconds(),
        #     num_chunks_streamed = num_chunks_streamed,
        #     system_fingerprint = None
        # )
        # update_data_store(data_instance)

with st.sidebar:
    show_session_cost_stats(tokens_pricing_unit=MISTRAL_TOKENS_PRICING_UNIT, prompt_tokens=st.session_state["mistral_prompt_tokens"], response_tokens=st.session_state["mistral_response_tokens"])
sidebar_footer()