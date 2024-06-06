# Getting started

## Set up Python environment
1. Install a conda distribution, such as [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation).
2. Create and activate a new environment with the required packages:
```bash
micromamba env create -f py-env.yaml
conda activate py-env
```

# Lessons

## Lesson 1: Type writer effect in Python
How can you get a response from an LLM model to appear as if it is being typed out? This lesson shows how to do this in Python.

If you are new to Python or want to go deep on how stream buffering works, the notebook in this session is for you.

## Lesson 2: So many APIs, so little time

## A streamlit app
[Streamlit makes it easy to build basic converational apps](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) in pure Python code. This lesson branches off of the linked tutorial to show how to use any LLM model to generate responses.