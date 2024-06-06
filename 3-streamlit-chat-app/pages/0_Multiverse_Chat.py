import os
from datetime import datetime
import gc
import time

import streamlit as st
from openai import OpenAI
import pandas as pd

from helpers.openai import *
from helpers.cohere import *
from helpers.mistral import *
from helpers.llama import *
from helpers.common import *

### START PAGE CONTENT ### 
st.set_page_config(page_title="Multiverse Chat", page_icon="🤖", layout="wide")

with st.sidebar:
    openai_api_key = get_openai_key()
    cohere_api_key = get_cohere_key()
    mistral_api_key = get_mistral_key()

