from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core import LlamaManager, REFLECTOR_GRAMMAR

@dataclass
class ReflectorOutput:
    thought: str
    finished: bool

class Reflector():
    """
    A simple agent that chooses whether the agent should continue working or not.
    """
    def __init__(
        self,
        manager: LlamaManager,
        system_prompt: str,
    ):
        self.manager = manager
        # TODO: construct system prompt for reflector
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.grammar = REFLECTOR_GRAMMAR

    def reflect(self, raw_text: str, message_history: List[Dict] = []) -> ReflectorOutput:
        # TODO: design this interface
        pass