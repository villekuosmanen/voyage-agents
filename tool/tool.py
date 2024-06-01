from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    name: str
    description: str


    def __init__(self) -> None:
        pass

    @abstractmethod
    def use(self, **kwargs):
        pass

class SearchObjectsTool(Tool):
    name: str = "SEARCH_OBJECTS"
    description: str = "TODO"

    def __init__(self) -> None:
        pass

    def use(self, object_name: str):
        pass
