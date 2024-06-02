from typing import List
import inspect

from tool import Tool

tool_explanation = """
You do not answer to the user directly - rather, your job is to choose what tool should be used to get the answer. You should analyse the problem first before choosing what tool to use.
"""

chain_of_thought_explanation = """
Before you write your answer, analyse the problem and what tool would be useful. If you think the query can be answered instantly, or can't be answered with any of these tools, simply answer with PASS.
If one of the provided tools can be used to answer the question, answer with "TOOL <command> <args>". The output from the command as well as the original query will then be routed to another support agent.
"""

reflector_explanation = """
You are analysing the response of a tool used by an AI agent to assist the use with their query, as well as notes from the previous AI agent.
Your task is to determine whether the user's query has been successfully actioned, or the question they posed can now be successfully answered.
"""

chain_of_thought_reflector_explanation = """
Before you write your answer, describe what action the previous agent took, and whether the task is not complete. Your analysis will be used by the tool calling agent to determine what tool to use next.
Based on your analysis, set the finished status as true or false.
"""

agent_start_prompt = """
You are coordinating a group of agents in helping with the given task or question, which may require some tools to complete.
Describe with a high level plan how the user's request can be serviced, referencing the list of tools where necessary.
"""

def construct_agent_system_prompt(prompt: str, tools: List[Tool]) -> str:
    prompt_lines = [
        prompt,
        agent_start_prompt,
        "TOOLS:",
        get_tools_prompt(tools),
    ]
    return '\n'.join(prompt_lines)

def construct_system_prompt(prompt: str, tools: List[Tool]) -> str:
    prompt_lines = [
        prompt,
        tool_explanation,
        "TOOLS:",
        get_tools_prompt(tools),
        chain_of_thought_explanation,
    ]
    return '\n'.join(prompt_lines)

def get_tools_prompt(tools: List[Tool]) -> str:
    tool_lines = []
    for tool in tools:
        tool_name = tool.name
        doc = inspect.getdoc(tool)
        use_method = tool.call
        sig = inspect.signature(use_method)
        params = sig.parameters
        
        args = []
        for param in params.values():
            if param.name == 'self':
                continue
            arg_name = param.name
            args.append(f"<{arg_name}>")
        
        tool_args = ' '.join(args)
        tool_lines.append(f"- {tool_name} {tool_args}\n{add_indentation(doc)}")

    return '\n'.join(tool_lines)


def add_indentation(docstring, indentation='    - '):
    lines = docstring.split('\n')
    indented_lines = [f"{indentation}{line}" for line in lines]
    return '\n'.join(indented_lines)

def construct_reflector_prompt(prompt: str):
    prompt_lines = [
        prompt,
        reflector_explanation,
        chain_of_thought_reflector_explanation,
    ]
    return '\n'.join(prompt_lines)
