from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import streamlit as st
from currency_converter import CurrencyConverter
from transformers import AutoTokenizer
from mistralai.models.chat_completion import ChatMessage
import time 
import os
 
BOT_OPENING_LINE = "sup?"
META_PROMPT = "Say something to the bot..."
USER_ROLE = "user"
LM_ROLE = "assistant"
SYSTEM_ROLE = "system"
DEFAULT_MODEL = "mistral-tiny"
MISTRAL_TOKENS_PRICING_UNIT = 1000000
TOKENIZE_STRING_ADJUSTMENT = -1
TOKENIZE_HISTORY_BUFFER = 2
CHAT_HISTORY_ADJUSTMENT_PER_MESSAGE = 4

c = CurrencyConverter()
def currency_converstion(euros, target="USD"):
    return c.convert(euros, 'EUR', target)

MODEL_DATA = {
    'mistral-tiny': {'input': 0.14, 'output': .42},
    'mistral-small': {'input': .6, 'output': 1.8},
    'mistral-medium': {'input': 2.5, 'output': 7.5}
}
MODEL_DATA = {
    k: {
        "input": currency_converstion(v["input"]),
        "output": currency_converstion(v["output"]),
    }
    for k, v in MODEL_DATA.items()
}
model_name_tokenizer_lookup = {
    "mistral-tiny": "mistralai/Mistral-7B-v0.1",
    "mistral-small": "mistralai/Mixtral-8X7B-v0.1",

    # assumption: closed model without public tokenizer, as far as I'm aware.
    "mistral-medium": "mistralai/Mixtral-8X7B-v0.1", 
}
PROVIDER = "Mistral"

os.environ['TOKENIZERS_PARALLELISM']="false"

def get_mistral_key():
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
    return mistral_api_key

def get_icon(role):
    if role == USER_ROLE:
        return st.session_state.user_icon
    elif role == LM_ROLE:
        return st.session_state.lm_icon
    else:
        raise ValueError(f"Role {role} not recognized.")

class TokenizersEstimateInstance(BaseModel):
    time: datetime
    provider: str
    api_id: str
    messages: List[Dict[str, str]]
    response: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    actual_prompt_tokens: int
    actual_request_tokens: int

def num_tokens_from_string(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string for the active OpenAI model.
    https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    encoding = AutoTokenizer.from_pretrained(model_name_tokenizer_lookup[model_name])
    num_tokens = len(encoding.encode(string)) + TOKENIZE_STRING_ADJUSTMENT
    return num_tokens

def num_tokens_from_prompt(messages: List[ChatMessage], model_name:str) -> int:
    """
    Returns the number of tokens used by a message and chat history.
    The Cohere docs aren't very specific about how to count tokens.
    https://docs.cohere.com/reference/tokens
    """

    num_tokens = TOKENIZE_HISTORY_BUFFER 
    for msg in messages:
        num_tokens += num_tokens_from_string(msg.content, model_name) + CHAT_HISTORY_ADJUSTMENT_PER_MESSAGE
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
        st.session_state["mistral_model"] = st.selectbox(
            "Select a model",
            MODEL_DATA.keys(),
            index=list(MODEL_DATA.keys()).index(st.session_state["mistral_model"])
        )
        return system_msg

        # model_info = MODEL_DATA[st.session_state["openai_model"]]
        # temperature = st.slider('Temperature', min_value=0., max_value=2., value=1., step=0.01) 
        # max_tokens = st.slider('Maximum length', min_value=1, max_value=model_info['max_tokens'], value=model_info['max_tokens']//2, step=1)
        # stop_sequences = st.text_input("Stop sequences - space separated list <= 4", "")
        # stop_sequences = stop_sequences.split(" ")[:4]
        # if stop_sequences == ['']:
        #     stop_sequences = None
        # top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=0.01)
        # frequency_penalty = st.slider('Frequency penalty', min_value=0., max_value=2., value=0., step=0.01)
        # presence_penalty = st.slider('Presence penalty', min_value=0., max_value=2., value=0., step=0.01)
        # return system_msg, frequency_penalty, max_tokens, presence_penalty, stop_sequences, temperature, top_p