from typing import Any, Dict, List, Optional, Tuple

from core import LlamaManager, generate_grammar
from agents import Reflector, ToolCaller
from tool import Tool
from prompt import construct_system_prompt

success_prompt = "Great job, the agent reported that the task completed successfully. Using the message history of the conversation, leave a short summary for the user describing the actions taken."
failure_prompt = "Unfortunately the agent reported that the task did not complete successfully. Using the message history of the conversation, leave a short summary for the user describing the actions taken."

class Agent():
    """
    Higher level agent capable of delegating tasks to individual ToolCallers.
    
    Logic:
    - original prompt given to tool caller.
    - reflector comments on tool output and decides whether the task should continue or not.
    - agent's number of loops setting chooses if the action is allowed to continue
    - if continues, agentic output is added to system log and process restarts
    """
    def __init__(
            self,
            manager: LlamaManager,
            tools: List[Tool],
            system_prompt: str,
            max_iterations: int = 10,
        ) -> None:
        self.manager = manager
        self.tool_caller = ToolCaller(manager, tools, system_prompt)
        self.reflector = Reflector(manager, system_prompt)
        self.max_iterations = max_iterations

    def run(self, raw_data, message_history: List[Dict] = []):
        did_complete, message_history = self._run_agent(raw_data, message_history)
        if did_complete:
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": success_prompt}
                ]
            })
        else:
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": failure_prompt}
                ]
            })

        return self.summarise()

    def summarise(self, message_history: List[Dict]) -> str:
        return self.manager.query(message_history)
        
    def _run_agent(self, raw_data, message_history: List[Dict] = []) -> Tuple[bool, List[Dict]]:
        message_history.append({
                "role": "user",
                "content": [
                    {"type" : "text", "text": raw_data},
                ]
            })

        i = 0
        while i < self.max_iterations:
            res = self.tool_caller.call(message_history=message_history)
            if res.success == False:
                return False, message_history
        
            # add to conversation
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": res.output}
                ]
            })

            res = self.reflector.reflect(message_history=message_history)
            if res.finished:
                return True, message_history
            
            # add to conversation
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": res.thought}
                ]
            })
