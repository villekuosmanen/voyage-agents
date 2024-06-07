# voyage-agents
AI Agents for local GPUs, by Voyage Robotics.

Originally built for a hackathon in 24 hours.

Find a 7B param multi-modal LLaVa x Mistral model in [HuggingFace](https://huggingface.co/villekuosmanen/LLaVa-1.6-Mistral-7B-llama.cpp/tree/main). You can also build models yourself using the llama.cpp library.

This is a standalone package. You need to install the [llama_cpp python bindings](https://github.com/abetlen/llama-cpp-python) as well to use voyage_agents.

## Installation

See the `llama-cpp-python` install instructions in the [README](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation) as the process varies based on your machine configuration, e.g. whether to use CUDA or not.

```
pip install llama-cpp-python
pip install voyage_agents
```

## Example usage

Imports and model creation.

```python3
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

from voyage_agents.core import LlamaManager

system_prompt = "You are an intelligent robotic assistant that is navigating autonomously in an environment, attempting to complete a given task, and assisting a user with queries."
model_path = "data/llava-v1.6-mistral-7b/llava-v1.6-mistral-7b-8B-F32.gguf"
clip_model_path="data/llava-v1.6-mistral-7b/mmproj.bin"

chat_handler = Llava16ChatHandler(clip_model_path=clip_model_path)
llama = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_gpu_layers=20, # Uncomment to use GPU acceleration
    n_ctx=4092, # Uncomment to increase the context window
    seed=1337, # Uncomment to set a specific seed
)
manager = LlamaManager(llama, temperature=0)
```

Calling a tool based on a prompt, and returning the result in structured form.

```python3
from voyage_agents.agents import ToolCaller
from voyage_agents.tool import SearchObjectsTool, PickObjectTool, ChangeTaskTool

tool_caller = ToolCaller(manager, [SearchObjectsTool(), PickObjectTool(), ChangeTaskTool()], system_prompt)

# easy task
print(tool_caller.call("have you found the object you are looking for yet?"))
```

Answering a question with the use of tools.

```python3
from voyage_agents.agents import QuestionAnswerer
from voyage_agents.tool import SearchObjectsTool, PickObjectTool, ChangeTaskTool

agent = QuestionAnswerer(manager, [SearchObjectsTool(), PickObjectTool(), ChangeTaskTool()], system_prompt)

# easy task
answer = agent.run("how many tables has the robot seen so far?")
print(answer)
```

See the `tools` package for examples on how you'd implement your own tools.
