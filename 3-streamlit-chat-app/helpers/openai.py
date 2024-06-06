from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import tiktoken
import streamlit as st
import time
import os
 
BOT_OPENING_LINE = "sup?"
META_PROMPT = "Say something to the bot..."
USER_ROLE = "user"
LM_ROLE = "assistant"
SYSTEM_ROLE = "system"
DEFAULT_MODEL = "gpt-3.5-turbo-1106"
OPENAI_TOKENS_PRICING_UNIT = 1000

# TODO: dynamically look up prices by scraping https://openai.com/pricing
MODEL_DATA = {
    'gpt-3.5-turbo-1106': {'input': 0.0010, 'output': 0.0020, 'max_tokens': 4095},
    # 'gpt-3.5-turbo-1106-instruct': {'input': 0.0015, 'output': 0.0020, 'max_tokens': 4095},
    'gpt-4': {'input': 0.03, 'output': 0.06, 'max_tokens': 8191},
    'gpt-4-32k': {'input': 0.06, 'output': 0.12, 'max_tokens': 32767},
    'gpt-4-1106-preview': {'input': 0.01, 'output': 0.03, 'max_tokens': 4095},
}
PROVIDER = "OpenAI"

def get_openai_key():
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
    return openai_api_key

def get_icon(role):
    if role == USER_ROLE:
        return st.session_state.user_icon
    elif role == LM_ROLE:
        return st.session_state.lm_icon
    else:
        raise ValueError(f"Role {role} not recognized.")

class TiktokenEstimateInstance(BaseModel):
    time: datetime
    provider: str
    api_id: str
    messages: List[Dict[str, str]]
    response: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    actual_prompt_tokens: int
    actual_request_tokens: int
    system_fingerprint: Optional[str] 

def num_tokens_from_string(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string for the active OpenAI model.
    https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_prompt(messages: List[Dict[str, str]], model_name:str) -> int:
    """
    Returns the number of tokens used by a list of messages.
    The OpenAI docs imply this changes model to model, and aren't very specific about how to count tokens.
    https://platform.openai.com/docs/guides/text-generation/managing-tokens

    This cookbook is much more helpful, and is followed here:
    https://github.com/openai/openai-cookbook/blob/6c00ce2ff597f39aa5f636040d7832a597784fd8/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # this hardcoded list is icky... 
    # someday OpenAI might provide this data in an API so the cost tracking part of this app is superfluous
    if model_name in [
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613"
    ]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_prompt(messages, model_name="gpt-3.5-turbo-0613")
    elif "gpt-4" in model_name:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_prompt(messages, model_name="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_prompt() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def api_params_panel():

    with st.sidebar:
        st.divider()
        # create system message input
        st.markdown("#### System")
        system_msg = st.text_area(
            "Type system messages here.",
            "You are a helpful assistant."
        )

        # create model params selectors
        st.markdown("#### Model")
        st.session_state["openai_model"] = st.selectbox(
            "Select a model",
            MODEL_DATA.keys(),
            index=list(MODEL_DATA.keys()).index(st.session_state["openai_model"])
        )
        model_info = MODEL_DATA[st.session_state["openai_model"]]
        temperature = st.slider('Temperature', min_value=0., max_value=2., value=1., step=0.01) 
        max_tokens = st.slider('Maximum length', min_value=1, max_value=model_info['max_tokens'], value=model_info['max_tokens']//2, step=1)
        stop_sequences = st.text_input("Stop sequences - space separated list <= 4", "")
        stop_sequences = stop_sequences.split(" ")[:4]
        if stop_sequences == ['']:
            stop_sequences = None
        top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=0.01)
        frequency_penalty = st.slider('Frequency penalty', min_value=0., max_value=2., value=0., step=0.01)
        presence_penalty = st.slider('Presence penalty', min_value=0., max_value=2., value=0., step=0.01)
        return system_msg, frequency_penalty, max_tokens, presence_penalty, stop_sequences, temperature, top_p