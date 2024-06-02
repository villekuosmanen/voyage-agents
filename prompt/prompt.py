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

def construct_system_prompt(prompt: str, tools: List[Tool]) -> str:
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

    tools_prompt = '\n'.join(tool_lines)
    prompt_lines = [
        prompt,
        tool_explanation,
        "TOOLS:",
        tools_prompt,
        chain_of_thought_explanation,
    ]
    return '\n'.join(prompt_lines)

def add_indentation(docstring, indentation='    - '):
    lines = docstring.split('\n')
    indented_lines = [f"{indentation}{line}" for line in lines]
    return '\n'.join(indented_lines)