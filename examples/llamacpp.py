from llama_cpp import Llama 
from llama_cpp.llama_chat_format import Llava16ChatHandler

system_prompt = """
You are a member of a helpful user support team for a robot that is navigating autonomously in an environment, attempting to complete a given task. You are assisting the user of the robot. The user is monitoring the robot remotely, and is also able to give certain commands to the robot.

Your job is to determine whether the query from the user can be answered instantly, or whether it requires extra data from the system in the form of one of the commands below. You don't actually answer the query - this is the job of another support agent. All you do is determine whether the question needs more data from one of the commands below to answer.

COMMANDS:
- SEARCH_OBJECTS <name>
    - searches the list of objects detected by the robot that match the given name.
    
Before you write your answer, analyse whether the question can be answered with PASS, or whether one of the above commands needs to be used. Afterwards, write your answer on a new line.

If you think the query can be answered instantly, simply answer with PASS. This should be your preferred option to use.
If more data is needed to answer the question, answer with "COMMAND <command> <args>". The output from the command as well as the original query will then be routed to another support agent.

Here's some example dialogues demonstrating successful replies to a user.

--- DEMONSTRATION ---
SYSTEM PROMPT: Robot's current task is now "move book from chest_of_drawers to table"
USER: what object is the robot currently looking for?
ANSWER: PASS
--- END OF DEMONSTRATION ---

--- DEMONSTRATION ---
SYSTEM PROMPT: Robot's current task is now "move book from bench to bookcase"
USER: how many books has the robot detected?
ANSWER: COMMAND SEARCH_OBJECTS book
--- END OF DEMONSTRATION ---

--- DEMONSTRATION ---
SYSTEM PROMPT: Robot's current task is now "move photo_frame from table to nightstand"
USER: has the robot found the nightstand yet?
ANSWER: COMMAND SEARCH_OBJECTS nightstand
--- END OF DEMONSTRATION ---

--- SYSTEM ACTION LOG ---

SYSTEM: using the following robot: Hello Robot Stretch 2
SYSTEM: robot's current task is now "move picture_frame from bench to table"

--- END OF SYSTEM ACTION LOG ---

"""



chat_handler = Llava16ChatHandler(clip_model_path="data/llava-v1.6-mistral-7b/mmproj.bin")
llm = Llama(
    model_path="data/llava-v1.6-mistral-7b/llava-v1.6-mistral-7b-8B-F32.gguf",
    chat_handler=chat_handler,
    n_gpu_layers=20, # Uncomment to use GPU acceleration
    n_ctx=4092, # Uncomment to increase the context window
    # seed=1337, # Uncomment to set a specific seed
)

res = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type" : "text", "text": "USER: have you found the object you are looking for yet?"},
            ]
        }
    ],
    response_format={ "type": "text" }
)
print()
print(res)
if len(res['choices']) == 1:
    print(res['choices'][0]['message']['content'])
