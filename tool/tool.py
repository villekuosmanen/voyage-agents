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
    description: str


    def __init__(self) -> None:
        pass

    @abstractmethod
    def call(self, **kwargs) -> ToolResult:
        pass

class SearchObjectsTool(Tool):
    name: str = "SEARCH_OBJECTS"
    description: str = "TODO"

    def __init__(self) -> None:
        pass

    def call(self, object_name: str) -> ToolResult:
        # TODO: implement this
        return ToolResult(True, f'ID: 2, name: {object_name}')
    
class PickObjectTool(Tool):
    name: str = "PICK_OBJECT"
    description: str = "TODO"

    def __init__(self) -> None:
        pass

    def call(self, object_id: int) -> ToolResult:
        # TODO: implement this
        return ToolResult(True, None)
    
class ChangeTaskTool(Tool):
    name: str = "CHANGE_TASK"
    description: str = "TODO"

    def __init__(self) -> None:
        pass

    def call(self, description: StringWithSpaces) -> ToolResult:
        # TODO: implement this - it would be a nested agent
        return ToolResult(True, description)
