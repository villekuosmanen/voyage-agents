from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava16ChatHandler

from core import LlamaManager

system_prompt = """
You are a member of a helpful user support team for a robot that is navigating autonomously in an environment, attempting to complete a given task. You are assisting the user of the robot. The user is monitoring the robot remotely, and is also able to give certain commands to the robot.

Your job is to determine whether the query from the user can be answered instantly, or whether it requires extra data from the system in the form of one of the commands below. You don't actually answer the query - this is the job of another support agent. All you do is determine whether the question needs more data from one of the commands below to answer.

COMMANDS:
- SEARCH_OBJECTS <name>
    - searches the list of objects detected by the robot that match the given name.
    
Before you write your answer, analyse whether the question can be answered with PASS, or whether one of the above commands needs to be used. Afterwards, write your answer on a new line.

If you think the query can be answered instantly, simply answer with PASS. This should be your preferred option to use.
If one of the provided tools can be used to answer the question, answer with "COMMAND <command> <args>". The output from the command as well as the original query will then be routed to another support agent.
--- SYSTEM ACTION LOG ---

SYSTEM: using the following robot: Hello Robot Stretch 2
SYSTEM: robot's current task is now "move picture_frame from bench to table"

--- END OF SYSTEM ACTION LOG ---
"""

model_path = "data/llava-v1.6-mistral-7b/llava-v1.6-mistral-7b-8B-F32.gguf"
clip_model_path="data/llava-v1.6-mistral-7b/mmproj.bin"

with open("examples/tools.gbnf","r") as f:
    grammar_string = f.read()
grammar = LlamaGrammar.from_string(grammar_string)

manager = LlamaManager(model_path, clip_model_path)
manager.start()

print(manager.query(system_prompt, "USER: have you found the object you are looking for yet?", grammar))
print(manager.query(system_prompt, "USER: how many coffee tables have you seen so far?", grammar))
print(manager.query(system_prompt, "USER: what is the robot trying to do?", grammar))
print(manager.query(system_prompt, "USER: What day is it?", grammar))

