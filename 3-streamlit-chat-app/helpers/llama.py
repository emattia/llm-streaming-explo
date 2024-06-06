import os
import signal
import subprocess
from threading import Thread
import re
import streamlit as st

BOT_OPENING_LINE = "sup?"
META_PROMPT = "Say something to the bot..."
USER_ROLE = "user"
LM_ROLE = "assistant"
SYSTEM_ROLE = "system"
DEFAULT_MODEL = "llama2"
PROVIDER="Llama.cpp & Ollama"
DEFAULT_PORT = 11434

def find_available_models(page_url = "https://ollama.ai/library"):
    from bs4 import BeautifulSoup
    import requests
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="repo")
    h2s = results.find_all("h2")
    models = [h2.get_text().strip() for h2 in h2s]
    return models

MODEL_DATA = {
    m: {} for m in find_available_models()
}

def get_icon(role):
    if role == USER_ROLE:
        return st.session_state.user_icon
    elif role == LM_ROLE:
        return st.session_state.lm_icon
    else:
        raise ValueError(f"Role {role} not recognized.")

class OllamaRunner(object):

    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _exec_cmd(self, cmd):
        return subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run(self):
        self.run_process = self._exec_cmd(["ollama", "run", self.model_name])

    def check_model(self):
        current_images = subprocess.run(['ollama', 'list'], check=True, stdout=subprocess.PIPE)
        current_images = current_images.stdout.decode('utf-8')
        current_image_data = [
            cell.strip()
            for cell in re.split("\\t|\\n", current_images)
            if cell != ""
        ]
        image_tags = current_image_data[::4][1:]
        model_names = [t.split(":")[0] for t in image_tags]
        return self.model_name in model_names

    def kill(self, rm_model=False):
        os.kill(self.run_process.pid, signal.SIGINT)
        if rm_model:
            self.kill_process = self._exec_cmd(["ollama", "rm", self.model_name])