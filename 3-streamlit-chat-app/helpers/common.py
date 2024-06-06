from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from typing import Optional
import streamlit as st
import plotly.express as px

COST_TRACKING_DATA_FILE = "cost_tracking.csv"

SAMPLE_PROMPTS = [
    "What are the core steps of designing and developing an API?",
    "What are the most impactful lessons of API design?",
    "What is the best way to learn data science?",
    "How many tokens of data do I need to fine-tune a BERT model?",
    "What is a more lucrative career path, learning how to build apps that use data science or optimizing CUDA kernels?"
    "Write 6 jokes with this level of dad-jokeness... What did the Buddhist ask the hot dog vendor? 'Make me one with everything.'",
    "Who is the best basketball player of all time?",
    "Does God exist? Explain the logic of your answer.",
    "What are the most influential frameworks for visualizing proofs in mathematics and philosophy?",
]

class DataInstance(BaseModel):
    time: datetime
    provider: str
    api_id: str
    prompt: str
    response: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_input_cost: float
    estimated_output_cost: float
    generation_time: float
    num_chunks_streamed: int
    system_fingerprint: Optional[str]  # Assuming this might be optional or not always present

def show_session_cost_stats(
    tokens_pricing_unit: int = None, 
    prompt_tokens: int = None,
    response_tokens: int = None
):

    with st.sidebar:
        st.divider()
        st.markdown(f"#### Session{' cost' if tokens_pricing_unit else ''} stats")

        st.text(f"Est. accumulated prompt tokens: {prompt_tokens}")
        st.text(f"Est. accumulated response tokens: {response_tokens}")

        if tokens_pricing_unit is not None:
            current_session_cost = round(
                (
                    prompt_tokens * st.session_state['prompt_cost'] + 
                    response_tokens * st.session_state['response_cost']
                ) / tokens_pricing_unit,
                2
            )
            st.text(f"Est. accumulated cost: {
                '$' + str(current_session_cost) 
                if current_session_cost > 0.00 
                else '< 1 cent'
            }")

        reset_chat = st.button("reset chat history")
        if reset_chat:
            st.session_state.messages = []

def sidebar_footer():
    with st.sidebar:
        st.markdown('### Made with streamlit')
        as_logo = open('icons/as-logo.svg').read()
        st.image(as_logo, width=200)

def update_data_store(data: DataInstance) -> None:
    new_df = pd.DataFrame([data.model_dump()], columns=list(DataInstance.model_fields.keys()))
    try: 
        df = pd.read_csv(COST_TRACKING_DATA_FILE)
        df = pd.concat([df, new_df])
    except:
        df = new_df
    df.to_csv(COST_TRACKING_DATA_FILE, index=False)

def make_3d_plot(df, showlegend=True):
    fig = px.scatter_3d(df, x='Input tokens', y='Output tokens', z='Generation time', color='Provider - API')
    fig.update_layout(plot_bgcolor = "#EFD1A8", showlegend=showlegend)
    return fig

def make_tokens_per_second_bar_plot(df, showlegend=True):
    fig = px.bar(df, x='Provider - API', y='Tokens per second', color='Provider - API', text='Tokens per second')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(plot_bgcolor = "#EFD1A8", showlegend=showlegend, xaxis={'categoryorder':'total descending'})
    return fig