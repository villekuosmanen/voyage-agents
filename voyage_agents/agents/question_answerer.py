from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from voyage_agents.core import LlamaManager, construct_system_message
from voyage_agents.agents import ToolCaller, ToolResult
from voyage_agents.tool import Tool

question_answerer_prompt = "Your very important job is to answer the user's question or query. Use the provided results from an action taken by a previous AI agent in your answer when necessary."

@dataclass
class QuestionResponse:
    response: str
    tool_result: Optional[ToolResult]

class QuestionAnswerer():
    """
    Single-round agent that uses a tool to answer a question.
    """
    def __init__(
            self,
            manager: LlamaManager,
            tools: List[Tool],
            system_prompt: str,
        ) -> None:
        self.manager = manager
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_caller = ToolCaller(manager, tools, system_prompt)

    def run(self, raw_data, message_history: List[Dict] = []) -> QuestionResponse:
        user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": raw_data}
                ]
            }
        message_history.append(user_message)

        # call the tool caller
        res = self.tool_caller.call(message_history=message_history)

        # add tool caller's thought into message history 
        message_history.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": res.thought}
            ]
        })
            
        tool_name = 'PASS'
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

        system_prompt = f"""
        {self.system_prompt}

        {question_answerer_prompt}
        """
        messages = [construct_system_message(system_prompt)] + message_history
        query_res = self.manager.query(messages)
        return QuestionResponse(query_res, res.tool_result)
