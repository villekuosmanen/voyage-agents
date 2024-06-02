from typing import Any, Dict, List, Optional, Tuple

from core import LlamaManager, construct_system_message
from agents import Reflector, ToolCaller
from tool import Tool
from prompt import construct_agent_system_prompt

success_prompt = "Great job, the agent reported that the task completed successfully. Referencing the message history when necessary, answer to the user in the most appropriate way."
failure_prompt = "Unfortunately the agent reported that the task did not complete successfully. Referencing the message history when necessary, answer to the user in the most appropriate way."

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
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_caller = ToolCaller(manager, tools, system_prompt)
        self.reflector = Reflector(manager, system_prompt)
        self.max_iterations = max_iterations

    def run(self, raw_data, message_history: List[Dict] = []):
        user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": raw_data}
                ]
            }
        messages = [
            construct_system_message(construct_agent_system_prompt(self.system_prompt, self.tools))
        ] + message_history + [user_message]
        res = self.manager.query(messages)
        message_history.append(user_message)
        message_history.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res}
                ]
            })

        did_complete, message_history = self._run_agent(message_history)
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

        return self.summarise(message_history)

    def summarise(self, message_history: List[Dict]) -> str:
        return self.manager.query(message_history)
        
    def _run_agent(self, message_history: List[Dict] = []) -> Tuple[bool, List[Dict]]:
        i = 0
        while i < self.max_iterations:
            res = self.tool_caller.call(message_history=message_history)
            if res.success == False:
                return False, message_history
        
            # add to conversation
            message_history.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res.thought}
                ]
            })
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f'{res.tool_used.name}: {res.output}'}
                ]
            })

            res = self.reflector.reflect(message_history=message_history)
            if res.finished:
                return True, message_history
            
            # add to conversation
            message_history.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res.thought}
                ]
            })
