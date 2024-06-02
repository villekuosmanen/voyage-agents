

class Agent():
    """
    Higher level agent capable of delegating tasks to individual ToolCallers.
    
    Logic:
    - original prompt given to tool caller.
    - reflector comments on tool output and decides whether the task should continue or not.
    - agent's number of loops setting chooses if the action is allowed to continue
    - if continues, agentic output is added to system log and process restarts
    """