from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

from core import LlamaManager
from agents import ToolCaller
from tool import SearchObjectsTool, PickObjectTool, ChangeTaskTool

system_prompt = """You are an intelligent robotic assistant that is navigating autonomously in an environment, attempting to complete a given task, and assisting a user with queries."""

system_action_log = """
--- SYSTEM ACTION LOG ---

SYSTEM: using the following robot: Hello Robot Stretch 2
SYSTEM: robot's current task is now "move picture_frame from bench to table"

--- END OF SYSTEM ACTION LOG ---
"""

model_path = "data/llava-v1.6-mistral-7b/llava-v1.6-mistral-7b-8B-F32.gguf"
clip_model_path="data/llava-v1.6-mistral-7b/mmproj.bin"


chat_handler = Llava16ChatHandler(clip_model_path=clip_model_path)
llama = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_gpu_layers=20, # Uncomment to use GPU acceleration
    n_ctx=4092, # Uncomment to increase the context window
    # seed=1337, # Uncomment to set a specific seed
)
manager = LlamaManager(llama)

tool_caller = ToolCaller(manager, [SearchObjectsTool(), PickObjectTool(), ChangeTaskTool()], system_prompt)
tool_caller.add_system_message(system_action_log)

# search objects
print(tool_caller.call("have you found the object you are looking for yet?"))
print(tool_caller.call("How many coffee tables have you seen so far?"))
# pick object
print(tool_caller.call("can you pick up the apple with id 32?"))
# change task
print(tool_caller.call("can you pick the apple from the table, and place it on the bed?"))
print(tool_caller.call("instead of the table, can you place the curent object on the shelf?"))
# other (pass)
print(tool_caller.call("what is the robot trying to do?"))
print(tool_caller.call("What day is it?"))
