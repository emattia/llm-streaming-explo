from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
from tokenizers import Tokenizer
import streamlit as st
import time
import os

BOT_OPENING_LINE = "sup?"
META_PROMPT = "Say something to the bot..."
USER_ROLE = "USER"
LM_ROLE = "CHATBOT"
SYSTEM_ROLE = "system"
DEFAULT_MODEL = "command"
COHERE_TOKENS_PRICING_UNIT = 1000000

# TODO: dynamically look up prices by scraping https://cohere.com/pricing
MODEL_DATA = {
    'command': {'input': 1.0, 'output': 2.0, 'max_tokens': 4095},
}
PROVIDER = "Cohere"

### TOKENIZER PARAMS BASED ON EMPIRICAL STUDY ### 
# There are some hard-coded assumptions about the number of tokens used by each message.
# These need further study with various Cohere parameters.
TOKENIZER = Tokenizer.from_pretrained("Cohere/command-nightly")
INPUT_BUFFER_TOKENS = 49
CHAT_HISTORY_ADJUSTMENT_PER_MESSAGE = -2
OUTPUT_BUFFER_TOKENS = -2

def get_cohere_key():
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
    return cohere_api_key

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
    message: str
    chat_history: List[Dict[str, str]]
    response: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    actual_prompt_tokens: int
    actual_request_tokens: int


def  num_tokens_from_string(string: str) -> int:
    return len(TOKENIZER.encode(string).ids)

def num_tokens_from_prompt(message: str, history: List[Dict[str, str]]) -> int:
    """
    Returns the number of tokens used by a message and chat history.
    The Cohere docs aren't very specific about how to count tokens.
    https://docs.cohere.com/reference/tokens
    """
    num_tokens = num_tokens_from_string(message) + sum([
        num_tokens_from_string(h["message"]) + CHAT_HISTORY_ADJUSTMENT_PER_MESSAGE
        for h in history
    ]) + INPUT_BUFFER_TOKENS 
    return num_tokens

def num_tokens_from_response(response: str) -> int:
    return num_tokens_from_string(response) + OUTPUT_BUFFER_TOKENS

def api_params_panel():

    with st.sidebar:
        st.divider()
        st.markdown("#### Model")
        st.session_state["cohere_model"] = st.selectbox(
            "Select a model",
            MODEL_DATA.keys(),
            index=list(MODEL_DATA.keys()).index(st.session_state["cohere_model"])
        )
        temperature = st.slider('Temperature', min_value=0., max_value=2., value=.3, step=0.01)
        return temperature 