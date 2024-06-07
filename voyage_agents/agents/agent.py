from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from voyage_agents.core import LlamaManager, construct_system_message
from voyage_agents.agents import Reflector, ToolCaller, ToolResult
from voyage_agents.tool import Tool
from voyage_agents.prompt import construct_agent_system_prompt

success_prompt = "Great job, the agent reported that the task completed successfully. Referencing the message history when necessary, answer to the user in the most appropriate way."
failure_prompt = "Unfortunately the agent reported that the task did not complete successfully. Referencing the message history when necessary, answer to the user in the most appropriate way."

@dataclass
class AgentResponse:
    response: str
    tool_results: List[Optional[ToolResult]]

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

    def run(self, raw_data, message_history: List[Dict] = []) -> AgentResponse:
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

        did_complete, message_history, tools_history = self._run_agent(message_history)
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

        return self.summarise(message_history, tools_history)

    def summarise(self, message_history: List[Dict], tools_history: List[Optional[ToolResult]]) -> str:
        response = self.manager.query(message_history)
        return AgentResponse(response, tools_history)
        
    def _run_agent(self, message_history: List[Dict] = []) -> Tuple[bool, List[Dict], List[Optional[ToolResult]]]:
        tool_results = []
        
        i = 0
        while i < self.max_iterations:
            res = self.tool_caller.call(message_history=message_history)
            if res.success == False:
                return False, message_history, tool_results
        
            # add to converstaion and tool results
            message_history.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res.thought}
                ]
            })
            tool_results.append(res.tool_result)

            tool_name = 'pass'
            tool_output = "<no output>"
            if res.tool_result is not None:
                tool_name = res.tool_result.tool_used.name
                tool_output = res.tool_result.textual_output
            if not res.success:
                # Make sure we mention tool failures to the agent.
                tool_output = "The use of this tool failed. No action was taken by this tool."
            
            message_history.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f'{tool_name}: {tool_output}'}
                ]
            })

            res = self.reflector.reflect(message_history=message_history)
            if res.finished:
                return True, message_history, tool_results
            
            # add to conversation
            message_history.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": res.thought}
                ]
            })
