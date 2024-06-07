from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ToolResponse:
    success: bool
    structured_output: Optional[Any]
    textual_output: Optional[Any]

class StringWithSpaces(str):
    pass

class Tool(ABC):
    name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def call(self, **kwargs) -> ToolResponse:
        pass

class SearchObjectsTool(Tool):
    """
    Query the detected object database for objects detected by the robot with the given name. Returns information about the objects, including their IDs.
    Does not physically search the environment for new objects.
    """
    name: str = "search_objects"

    def __init__(self) -> None:
        pass

    def call(self, object_name: str) -> ToolResponse:
        # TODO: implement this
        return ToolResponse(True, None, f'ID: 2, name: {object_name}')
    
class PickObjectTool(Tool):
    """
    Instructs the robot to pick an object with the given ID.
    """
    name: str = "pick_object"

    def __init__(self) -> None:
        pass

    def call(self, object_id: int) -> ToolResponse:
        # TODO: implement this
        return ToolResponse(True, None, "Pick object task queued successfully.")
    
class ChangeTaskTool(Tool):
    """
    Changes the current task of the robot. The input to this should be written in concise natural language, describing the objects involved and actions to take.
    This tool should be used for any actions that take more than one step to complete.
    """
    name: str = "change_task"

    def __init__(self) -> None:
        pass

    def call(self, description: StringWithSpaces) -> ToolResponse:
        # TODO: implement this - it would be a nested agent
        return ToolResponse(True, None, f'Task is now {description}')
