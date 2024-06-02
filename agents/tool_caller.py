from dataclasses import dataclass
import json
import shlex
from typing import Any, List, Optional

from core import LlamaManager, generate_grammar
from tool import Tool
from prompt import construct_system_prompt

@dataclass
class ToolCallResult:
    success: bool
    tool_used: Optional[Tool]
    output: Optional[Any]

class ToolCaller():
    def __init__(
            self,
            manager: LlamaManager,
            tools: List[Tool],
            system_prompt: str,
        ):
        self.manager = manager
        self.tools = {tool.name: tool for tool in tools}
        # TODO: construct prompt
        self.system_prompt = construct_system_prompt(system_prompt, tools)
        print(self.system_prompt)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        self.grammar = generate_grammar(tools)

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def call(self, raw_text: str) -> ToolCallResult:
        messages = self.messages.copy()
        messages.append({
                "role": "user",
                "content": [
                    {"type" : "text", "text": raw_text},
                ]
            })

        generated = self.manager.query(messages, self.grammar)
        res = json.loads(generated)
        print(res)
        command = res['command']

        # TODO: parse command
        if command == 'PASS':
            return ToolCallResult(True, None, None)
        
        tokens = shlex.split(command)
        assert tokens[0] == 'TOOL'

        tool = self.tools.get(tokens[1], None)
        assert tool is not None


        res = tool.call(*tokens[2:])
        return ToolCallResult(
            success=res.success,
            tool_used=tool,
            output=res.output
        )

