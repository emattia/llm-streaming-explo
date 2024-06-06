import os
from datetime import datetime
import gc
import time

import streamlit as st
from openai import OpenAI
import pandas as pd

from helpers.openai import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="OpenAI Chat", page_icon="🤖")

# LHS navigation
with st.sidebar:
    openai_api_key = get_openai_key()
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/1_OpenAI_Chat.py)"
    "[OpenAI Pricing data source](https://openai.com/pricing)"

if openai_api_key is None or openai_api_key == "":
    st.stop()
else:
    client = OpenAI(api_key=openai_api_key)
    del openai_api_key
    gc.collect()

# create a session state to log stuff
if "openai_model" not in st.session_state or "openai_prompt_tokens" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://openai.com/pricing to find the latest models and update MODEL_DATA in helpers/openai.py""")
    st.session_state["openai_model"] = DEFAULT_MODEL
    st.session_state["openai_prompt_tokens"] = 0
    st.session_state["openai_response_tokens"] = 0
    st.session_state["prompt_cost"] = MODEL_DATA[DEFAULT_MODEL]["input"]
    st.session_state["response_cost"] = MODEL_DATA[DEFAULT_MODEL]["output"]
    st.session_state.openai_messages = [{"role": LM_ROLE, "content": BOT_OPENING_LINE}]
    st.session_state.user_icon = open("icons/user.svg").read()
st.session_state.lm_icon = open("icons/chatgpt.svg").read()

# set up side bar
system_msg, frequency_penalty, max_tokens, presence_penalty, stop_sequences, temperature, top_p = api_params_panel()

# initialize chat history
if "openai_messages" not in st.session_state:
    st.session_state.openai_messages = [
        {"role": LM_ROLE, "content": BOT_OPENING_LINE}
    ]

# write out the openai_messages
for message in st.session_state.openai_messages:
    with st.chat_message(name=message["role"], avatar=get_icon(message["role"])):
        st.markdown(message["content"])

if client:

    if prompt := st.chat_input(META_PROMPT):

        # get user prompt
        with st.chat_message(name=USER_ROLE, avatar=get_icon(USER_ROLE)):
            st.markdown(prompt)

        # add prompt to chat history
        st.session_state.openai_messages.append({"role": USER_ROLE, "content": prompt})
        message_payload = [{"role": SYSTEM_ROLE, "content": system_msg}] + st.session_state.openai_messages

        # track token count
        n_prompt_tokens = num_tokens_from_prompt(message_payload, st.session_state["openai_model"])
        st.session_state["openai_prompt_tokens"] += n_prompt_tokens
        
        # chat iteration
        with st.chat_message(name=LM_ROLE, avatar=get_icon(LM_ROLE)):
            message_placeholder = st.empty()
            full_response = ""
            api_params = dict( 
                model=st.session_state["openai_model"], 
                messages=message_payload, 
                stream=True,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                temperature=temperature,
                top_p=top_p
            )
            num_chunks_streamed = 0; t0 = datetime.now()
            for response in client.chat.completions.create(**api_params):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + '|')
                num_chunks_streamed += 1
            tf = datetime.now()
            message_placeholder.markdown(full_response)

        # add response it to chat history
        st.session_state.openai_messages.append({"role": LM_ROLE, "content": full_response})

        # track token count
        n_response_tokens = num_tokens_from_string(full_response, st.session_state["openai_model"])
        st.session_state["openai_response_tokens"] += n_response_tokens

        data_instance = DataInstance(
            time = t0,
            provider = PROVIDER,
            api_id = st.session_state["openai_model"],
            prompt = prompt,
            response = full_response,
            estimated_input_tokens = n_prompt_tokens,
            estimated_output_tokens = n_response_tokens,
            estimated_input_cost = st.session_state['prompt_cost'] * n_prompt_tokens / OPENAI_TOKENS_PRICING_UNIT,
            estimated_output_cost = st.session_state['response_cost'] * n_response_tokens / OPENAI_TOKENS_PRICING_UNIT,
            generation_time = (tf - t0).total_seconds(),
            num_chunks_streamed = num_chunks_streamed,
            system_fingerprint = response.system_fingerprint
        )
        update_data_store(data_instance)

with st.sidebar:
    show_session_cost_stats(tokens_pricing_unit=OPENAI_TOKENS_PRICING_UNIT, prompt_tokens=st.session_state["openai_prompt_tokens"], response_tokens=st.session_state["openai_response_tokens"])
sidebar_footer()