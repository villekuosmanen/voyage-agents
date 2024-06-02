from typing import Any, Dict, List

from voyage_agents.core import LlamaManager, construct_system_message
from voyage_agents.agents import ToolCaller
from voyage_agents.tool import Tool

question_answerer_prompt = "Your very important job is to answer the user's question or query. Use the provided results from an action taken by a previous AI agent in your answer when necessary."

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

    def run(self, raw_data, message_history: List[Dict] = []):
        user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": raw_data}
                ]
            }
        message_history.append(user_message)

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

        system_prompt = f"""
        {self.system_prompt}

        {question_answerer_prompt}
        """
        messages = [construct_system_message(system_prompt)] + message_history
        return self.manager.query(messages)
