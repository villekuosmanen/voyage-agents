# voyage-agents
AI Agents for local GPUs, by Voyage Robotics.

Originally built for a hackathon in 24 hours.

Find a 7B param multi-modal LLaVa x Mistral model in [HuggingFace](https://huggingface.co/villekuosmanen/LLaVa-1.6-Mistral-7B-llama.cpp/tree/main). You can also build models yourself using the llama.cpp library.

This is a standalone package. You need to install llama_cpp as well to use voyage_agents.

## Installation

```
pip install llama-cpp-python
pip install voyage_agents
```

## Example usage

```python3
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

agent = QuestionAnswerer(manager, [SearchObjectsTool(), PickObjectTool(), ChangeTaskTool()], system_prompt)

# easy task
answer = agent.run("how many tables has the robot seen so far?")
print(answer)
```
