import ctypes

from llama_cpp import Llama, LlamaGrammar, llama_log_set, llama_log_callback
from llama_cpp.llama_chat_format import Llava16ChatHandler

@llama_log_callback
def log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    pass

class LlamaManager:
    def __init__(self, model_path, clip_model_path):
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        llama_log_set(log_callback, ctypes.c_void_p(0))

    def start(self):
        chat_handler = Llava16ChatHandler(clip_model_path=self.clip_model_path)
        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=chat_handler,
            n_gpu_layers=20, # Uncomment to use GPU acceleration
            n_ctx=4092, # Uncomment to increase the context window
            # seed=1337, # Uncomment to set a specific seed
        )

    def query(self, system_prompt, user_prompt, grammar):
        res = self.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": user_prompt},
                    ]
                }
            ],
            response_format={ "type": "text" },
            grammar=grammar,
        )
        return res['choices'][0]['message']['content']