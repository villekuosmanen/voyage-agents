import re

from PIL import Image
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

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
ASSISTANT: PASS
--- END OF DEMONSTRATION ---

--- DEMONSTRATION ---
SYSTEM PROMPT: Robot's current task is now "move book from bench to bookcase"
USER: how many books has the robot detected?
ASSISTANT: COMMAND SEARCH_OBJECTS book
--- END OF DEMONSTRATION ---

--- DEMONSTRATION ---
SYSTEM PROMPT: Robot's current task is now "move photo_frame from table to nightstand"
USER: has the robot found the nightstand yet?
ASSISTANT: COMMAND SEARCH_OBJECTS nightstand
--- END OF DEMONSTRATION ---

--- SYSTEM ACTION LOG ---

SYSTEM: using the following robot: Hello Robot Stretch 2
SYSTEM: robot's current task is now "move picture_frame from bench to table"

--- END OF SYSTEM ACTION LOG ---

"""

# TODO: replace "\_" with "_"

model_path = "llava-v1.6-mistral-7b"
conv_mode = "mistral_instruct"
temperature = 0.2

user_message = "have you found the object you are looking for yet?"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

conv = Conversation(
    system=system_prompt,
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
conv.append_message(conv.roles[0], user_message)

prompt = conv.get_prompt()
inputs = tokenizer([prompt])
input_ids = torch.as_tensor(inputs.input_ids).cuda()

with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        max_new_tokens=256,
        use_cache=True,
        grammar="lol"
    )
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()        
print(outputs)      
