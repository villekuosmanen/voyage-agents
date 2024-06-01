from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava16ChatHandler

from core import LlamaManager
from agents import ToolCaller
from tool import SearchObjectsTool

system_prompt = """
You are a member of a helpful user support team for a robot that is navigating autonomously in an environment, attempting to complete a given task. You are assisting the user of the robot. The user is monitoring the robot remotely, and is also able to give certain commands to the robot.

You should analyse the question first, then choose which tool can be used to answer the user's query.

TOOLS:
- SEARCH_OBJECTS <name>
    - searches the list of objects detected by the robot that match the given name.
    
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

tool_caller = ToolCaller(manager, [SearchObjectsTool()], system_prompt)

# with open("examples/tools.gbnf","r") as f:
#     grammar_string = f.read()
# grammar = LlamaGrammar.from_string(grammar_string)

manager.start()

print(tool_caller.call("USER: have you found the object you are looking for yet?"))
print(tool_caller.call("USER: how many coffee tables have you seen so far?"))
print(tool_caller.call("USER: what is the robot trying to do?"))
print(tool_caller.call("USER: What day is it?"))
