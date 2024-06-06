import streamlit as st
import pandas as pd
from helpers.common import *

st.set_page_config("LLM Chat Playground", layout="wide", initial_sidebar_state="expanded" )

try:    
    df = pd.read_csv(COST_TRACKING_DATA_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=list(DataInstance.model_fields.keys()))

st.markdown("# LLM Chat Playground")
st.markdown(
"""
## What is this application?
These pages and the [open-source code](https://github.com/alignment-systems/llm-streaming/blob/main/3-streamlit) will show you:

    - the basics of making LLM chat interfaces in a Python app, 
    - how to stream respones into `stdout` or an application, 
    - TODO: how to stream responses asynchonously,
    - how to estimate tokens, and therefore costs, for vendor APIs, _before_ sending requests.

This is a starter pack for Python devs focused on making it easy to start making chat loops with LLMs.
"""
)

st.markdown("""
         
## How to use this application
Pick your favorite LLM in the left-hand side navigation and start chatting.
As you start making requests, they will be tracked in a CSV file, visible in a dataframe on this page.
""")

st.markdown("## Cost tracking results")
st.markdown(f"""
When you use any of the chat applications, statisitics about the streaming response will be tracked.
"You can find the data that creates this table and the plots in `{COST_TRACKING_DATA_FILE}`.

If you want to scale this application up, you would probably want to store in an external database instead.
""")
st.dataframe(df)

if df.shape[0] > 0:
    grid = st.columns(2)
    _df = df.copy()
    _df['color_annotation'] = df['provider'] + ' - ' + df['api_id']
    _df = pd.DataFrame({
        'Input tokens': _df.estimated_input_tokens.values,
        'Output tokens': _df.estimated_output_tokens.values,
        'Generation time': _df.generation_time.values,
        'Provider - API': _df.color_annotation.values
    })
    _df['Tokens per second'] = _df['Output tokens'] / _df['Generation time']
    fig_3d = make_3d_plot(_df, showlegend=False)
    _df_avg = _df.groupby('Provider - API').mean().reset_index()
    fig_tokens_per_second = make_tokens_per_second_bar_plot(_df_avg)
    grid[0].plotly_chart(fig_3d, use_container_width=True)
    grid[1].plotly_chart(fig_tokens_per_second, use_container_width=True)

st.markdown("""
## Awesome text generation frontends 
If you find this project valuable, you will definitely learn a lot from interacting with these projects too:
- [Oobabooga's `text-generation-webui`](https://github.com/oobabooga/text-generation-webui)
- [Jeffrey Morgan and Michael Chiang's `ollama`](https://github.com/jmorganca/ollama?tab=readme-ov-file#community-integrations)
- [Nat Friedman's `openplayground`](https://github.com/nat/openplayground)
- [Nomic's `GPT4All`](https://gpt4all.io/index.html)
- [Iván Martínez' `PrivateGPT`](https://github.com/imartinez/privateGPT)
- [`LMStudio`](https://lmstudio.ai/)
- [`KoboldAI`](https://github.com/KoboldAI/KoboldAI-Client)
- [`Silly Tavern AI`](https://sillytavernai.com/)
""")

sidebar_footer()