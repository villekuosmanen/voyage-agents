import ctypes

from llama_cpp import Llama, llama_log_set, llama_log_callback

@llama_log_callback
def log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    pass

class LlamaManager:
    def __init__(self, llama: Llama):
        self.llm = llama
        llama_log_set(log_callback, ctypes.c_void_p(0))

    def query(self, messages, grammar):
        res = self.llm.create_chat_completion(
            messages=messages,
            response_format={ "type": "text" },
            grammar=grammar,
        )
        return res['choices'][0]['message']['content']