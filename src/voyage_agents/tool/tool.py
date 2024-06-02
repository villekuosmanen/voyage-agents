from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ToolResult:
    success: bool
    output: Optional[Any]

class StringWithSpaces(str):
    pass

class Tool(ABC):
    name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def call(self, **kwargs) -> ToolResult:
        pass

class SearchObjectsTool(Tool):
    """
    Query the detected object database for objects detected by the robot with the given name. Returns information about the objects, including their IDs.
    Does not physically search the environment for new objects.
    """
    name: str = "SEARCH_OBJECTS"

    def __init__(self) -> None:
        pass

    def call(self, object_name: str) -> ToolResult:
        # TODO: implement this
        return ToolResult(True, f'ID: 2, name: {object_name}')
    
class PickObjectTool(Tool):
    """
    Instructs the robot to pick an object with the given ID.
    """
    name: str = "PICK_OBJECT"

    def __init__(self) -> None:
        pass

    def call(self, object_id: int) -> ToolResult:
        # TODO: implement this
        return ToolResult(True, "Pick object task queued successfully.")
    
class ChangeTaskTool(Tool):
    """
    Changes the current task of the robot. The input to this should be written in concise natural language, describing the objects involved and actions to take.
    This tool should be used for any actions that take more than one step to complete.
    """
    name: str = "CHANGE_TASK"

    def __init__(self) -> None:
        pass

    def call(self, description: StringWithSpaces) -> ToolResult:
        # TODO: implement this - it would be a nested agent
        return ToolResult(True, f'Task is now {description}')
