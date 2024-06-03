from dataclasses import dataclass
import json
import shlex
from typing import Any, Dict, List, Optional

from voyage_agents.core import LlamaManager, generate_grammar, construct_system_message
from voyage_agents.tool import Tool
from voyage_agents.prompt import construct_system_prompt

@dataclass
class ToolResult:
    tool_used: Tool
    args: List[Any]
    structured_output: Optional[Any]
    textual_output: str

@dataclass
class ToolCallResult:
    thought: str
    success: bool
    tool_result: Optional[ToolResult]

class ToolCaller():
    """
    A simple agent capable for calling tools. Returns structured output.
    """
    def __init__(
            self,
            manager: LlamaManager,
            tools: List[Tool],
            system_prompt: str,
        ):
        self.manager = manager
        self.tools = {tool.name: tool for tool in tools}
        self.system_message = construct_system_message(construct_system_prompt(system_prompt, tools))
        self.grammar = generate_grammar(tools)

    def call(self, raw_text: Optional[str] = None, message_history: Optional[List[Dict]]= None) -> ToolCallResult:
        messages = [self.system_message] 
        if message_history is not None:
            messages += message_history
        if raw_text is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type" : "text", "text": raw_text},
                ]
            })
        generated = self.manager.query(messages, self.grammar)
        res = json.loads(generated)
        print(res)
        thought=res['thought']
        command = res['command']

        # TODO: parse command
        if command == 'PASS':
            return ToolCallResult(thought, True, None)
        
        tokens = shlex.split(command)
        assert tokens[0] == 'TOOL'

        tool = self.tools.get(tokens[1], None)
        assert tool is not None


        res = tool.call(*tokens[2:])
        return ToolCallResult(
            thought=thought,
            success=res.success,
            tool_result=ToolResult(
                tool_used=tool,
                args=tokens[2:],
                structured_output=res.structured_output,
                textual_output=res.textual_output,
            ),
        )

