from dataclasses import dataclass
from typing import Any, List, Tuple

from core import LlamaManager, generate_grammar
from tool import Tool

@dataclass
class Result:
    success: bool
    output: Any

class ToolCaller():
    def __init__(
            self,
            manager: LlamaManager,
            tools: List[Tool],
            system_prompt: str,
        ):

        print(tools)
        self.manager = manager
        self.tools = tools
        # TODO: construct prompt
        self.system_prompt = system_prompt
        self.grammar = generate_grammar(tools)

    def call(self, raw_text: str) -> Tuple[str, Result]:
        return self.manager.query(self.system_prompt, raw_text, self.grammar)


