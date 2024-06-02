from dataclasses import dataclass
import json
import shlex
from typing import Any, Dict, List, Optional

from core import LlamaManager, generate_grammar, construct_system_message
from tool import Tool
from prompt import construct_system_prompt

@dataclass
class ToolCallResult:
    thought: str
    success: bool
    tool_used: Optional[Tool]
    output: Optional[Any]

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

    def call(self, raw_text: Optional[str], message_history: Optional[List[Dict]]) -> ToolCallResult:
        messages = [self.system_message] 
        if message_history is not None:
            messages.append(message_history)
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
            return ToolCallResult(True, None, None, None)
        
        tokens = shlex.split(command)
        assert tokens[0] == 'TOOL'

        tool = self.tools.get(tokens[1], None)
        assert tool is not None


        res = tool.call(*tokens[2:])
        return ToolCallResult(
            thought=thought,
            success=res.success,
            tool_used=tool,
            output=res.output
        )

