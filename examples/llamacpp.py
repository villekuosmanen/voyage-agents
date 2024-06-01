from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava16ChatHandler

from core import LlamaManager
from agents import ToolCaller
from tool import SearchObjectsTool, PickObjectTool, ChangeTaskTool

system_prompt = """
You are an intelligent robotic assistant that is navigating autonomously in an environment, attempting to complete a given task. You are assisting a user with queries, and have access to a variety of tools to do so.

You should analyse the question first, then choose which tool can be used to answer the user's query.

TOOLS:
- SEARCH_OBJECTS <object_name>
    - searches objects detected by the robot for any that match the given name.
- PICK_OBJECT <object_id>
    - instructs the robot to pick an object with the given ID.
- CHANGE_TASK <description>
    - changes the current task of the robot. The input to this should be written in concise natural language, describing the objects involved and actions to take. 
    
Before you write your answer, analyse the problem and what tool would be useful. If you think the query can be answered instantly, or can't be answered with any of these tools, simply answer with PASS.
If one of the provided tools can be used to answer the question, answer with "TOOL <command> <args>". The output from the command as well as the original query will then be routed to another support agent.

--- SYSTEM ACTION LOG ---

SYSTEM: using the following robot: Hello Robot Stretch 2
SYSTEM: robot's current task is now "move picture_frame from bench to table"

--- END OF SYSTEM ACTION LOG ---
"""

model_path = "data/llava-v1.6-mistral-7b/llava-v1.6-mistral-7b-8B-F32.gguf"
clip_model_path="data/llava-v1.6-mistral-7b/mmproj.bin"
manager = LlamaManager(model_path, clip_model_path)

tool_caller = ToolCaller(manager, [SearchObjectsTool(), PickObjectTool(), ChangeTaskTool()], system_prompt)

manager.start()

# search objects
# print(tool_caller.call("USER: have you found the object you are looking for yet?"))
# print(tool_caller.call("USER: how many coffee tables have you seen so far?"))
# # pick object
# print(tool_caller.call("USER: can you pick up the apple with id 32?"))
# change task
print(tool_caller.call("USER: can you pick the apple from the table, and place it on the bed?"))
print(tool_caller.call("USER: instead of the table, can you place the curent object on the shelf?"))
# other (pass)
print(tool_caller.call("USER: what is the robot trying to do?"))
print(tool_caller.call("USER: What day is it?"))
