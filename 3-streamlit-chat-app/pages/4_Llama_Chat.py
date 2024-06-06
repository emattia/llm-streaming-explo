import os
from datetime import datetime
import shutil
import time
import json
import requests

import streamlit as st
import pandas as pd

from helpers.llama import *
from helpers.common import *


def run_model():
    # create process runner
    st.session_state.runner = OllamaRunner(model_name=st.session_state["llama_model"])
    
    # wait for model to be ready
    with st.spinner("Waiting for model to be ready..."):
        while not st.session_state.runner.check_model():
            time.sleep(2)
    

### START PAGE CONTENT ### 
st.set_page_config(page_title="Llama Chat", page_icon="🤖")

# look for ollama local path
if ollama_path := shutil.which("ollama") == "":
    st.text("ollama not found in PATH. Please [download ollama](https://ollama.ai/) and try again.")
    st.stop()

# create a session state to log stuff
if "llama_model" not in st.session_state or "ollama_prompt_tokens" not in st.session_state:
    if DEFAULT_MODEL not in MODEL_DATA:
        raise ValueError(f"""
                         Model {DEFAULT_MODEL} not found in pricing data. 
                         Go to https://ollama.ai/library to find the latest models and update MODEL_DATA in helpers/llama.py""")
    st.session_state["llama_model"] = DEFAULT_MODEL
    st.session_state["ollama_prompt_tokens"] = 0
    st.session_state["ollama_response_tokens"] = 0
    st.session_state.llama_messages = [{"role": LM_ROLE, "content": BOT_OPENING_LINE}]
    st.session_state.user_icon = open("icons/user.svg").read()
    run_model()

st.session_state.lm_icon = open("icons/llama.svg").read()

# Set up the model
with st.sidebar:
    "[This page's source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit/pages/4_Llama_Chat.py)"
    "[Ollama](https://ollama.ai/)"
    "[Llama.cpp](https://github.com/ggerganov/llama.cpp)"

    st.divider()

    st.markdown("## Model")
    st.session_state["llama_model"] = st.selectbox("Pick a model. Some in this list may be too big on a laptop. `llama2` and `mistral` are most reliable. See Ollama docs for details.", options=list(MODEL_DATA.keys()))
    port = DEFAULT_PORT
    if st.session_state["llama_model"] != st.session_state.runner.model_name:
        st.session_state.runner.kill()
        run_model()

    st.markdown(f"Model server on pid: {st.session_state.runner.run_process.pid}, port: {port}")


# TODO - let user choose port
url=f"http://localhost:{port}/api/chat"
temperature = 0.3
# initialize chat history
if "llama_messages" not in st.session_state:
    st.session_state.llama_messages = [
        {"role": LM_ROLE, "content": BOT_OPENING_LINE}
    ]

# write out the llama_messages
for message in st.session_state.llama_messages:
    with st.chat_message(name=message["role"], avatar=get_icon(message["role"])):
        st.markdown(message["content"])

if prompt := st.chat_input(META_PROMPT):

    # get user prompt
    with st.chat_message(name=USER_ROLE, avatar=get_icon(USER_ROLE)):
        st.markdown(prompt)

    # add prompt to chat history
    st.session_state.llama_messages.append({"role": USER_ROLE, "content": prompt})

    # chat iteration
    with st.chat_message(name=LM_ROLE, avatar=get_icon(LM_ROLE)):
        message_placeholder = st.empty()
        # full_response = ""
        full_response = "**" + st.session_state["llama_model"] + "**: "
        api_params = dict( 
            model=st.session_state["llama_model"], 
            messages=st.session_state.llama_messages, 
            stream=True,
            temperature=temperature,
        )
        num_chunks_streamed = 0; t0 = datetime.now()
        with requests.post(url, json=api_params, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                body = json.loads(line)
                if "error" in body:
                    raise Exception(body["error"])
                if body.get("done") is False:
                    message = body.get("message", "")
                    content = message.get("content", "")
                    full_response += content
                    message_placeholder.markdown(full_response + '|')
                if body.get("done", False):
                    message["content"] = full_response
                    message["n_prompt_tokens"] = body.get("prompt_eval_count", 0)
                    message["n_response_tokens"] = body.get("eval_count", 0)
                num_chunks_streamed += 1
        tf = datetime.now()
        message_placeholder.markdown(full_response)

        # add response it to chat history
        st.session_state.llama_messages.append({"role": LM_ROLE, "content": full_response})

        # track token count
        st.session_state["ollama_prompt_tokens"] += message["n_prompt_tokens"]
        st.session_state["ollama_response_tokens"] += message["n_response_tokens"]
        print(
            "Cumulative prompt tokens:", st.session_state["ollama_prompt_tokens"],
            "\nCumulative response tokens:", st.session_state["ollama_response_tokens"]
        )

        data_instance = DataInstance(
            time = t0,
            provider = PROVIDER,
            api_id = st.session_state["llama_model"],
            prompt = prompt,
            response = full_response,
            estimated_input_tokens = message["n_prompt_tokens"],
            estimated_output_tokens = message["n_response_tokens"],
            estimated_input_cost = 0.0,
            estimated_output_cost = 0.0,
            generation_time = (tf - t0).total_seconds(),
            num_chunks_streamed = num_chunks_streamed,
            system_fingerprint = None
        )
        update_data_store(data_instance)

with st.sidebar:
    show_session_cost_stats(prompt_tokens=st.session_state["ollama_prompt_tokens"], response_tokens=st.session_state["ollama_response_tokens"])
sidebar_footer()