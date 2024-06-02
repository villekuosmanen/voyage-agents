from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from voyage_agents.core import LlamaManager, REFLECTOR_GRAMMAR, construct_system_message
from voyage_agents.prompt import construct_reflector_prompt

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
        self.system_message = construct_system_message(construct_reflector_prompt(system_prompt))
        self.grammar = REFLECTOR_GRAMMAR

    def reflect(self, raw_text: str = None, message_history: List[Dict] = []) -> ReflectorOutput:
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

        return ReflectorOutput(
            thought=res['thought'],
            finished=res['finished']
        )
