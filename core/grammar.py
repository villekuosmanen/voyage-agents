from typing import List
import inspect

from llama_cpp import LlamaGrammar

from tool import Tool, StringWithSpaces

starting_grammar_rules = [
    'root ::= "{ \\"thought\\": " thought ", \\"command\\": \\"" (toolCall | passCall) "\\" }"',
    'thought ::= "\\"I think" [a-zA-Z0-9_ ]+ "\\""',
    'passCall ::= "PASS"',
    'stringWithSpacesArg ::= "\'" [a-zA-Z0-9_ ]+ "\'"',
    'stringArg ::= [a-zA-Z0-9_]+',
    'intArg ::= [0-9]+',
]

def generate_grammar(tools: List[Tool]):
    tool_names = []
    tool_rules = []

    for tool in tools:
        tool_command_name = tool.__class__.__name__
        tool_name = tool.name
        use_method = tool.call
        sig = inspect.signature(use_method)
        params = sig.parameters

        args = []
        print(params)
        for param in params.values():
            if param.name == 'self':
                continue
            # arg_name = param.name
            arg_type = param.annotation
            if arg_type == str:
                arg_type_str = "stringArg"
            elif arg_type == StringWithSpaces:
                arg_type_str = "stringWithSpacesArg"
            elif arg_type == int:
                arg_type_str = "intArg"
            else:
                raise ValueError(f'unknown argument type: {arg_type}')
            args.append(arg_type_str)

        # Generate grammar line for the tool
        arg_list = ' '.join(args)
        grammar_line = f'{tool_command_name} ::= "{tool_name} " {arg_list}'
        tool_rules.append(grammar_line)
        # TODO add a reference to command entry
        tool_names.append(tool_command_name)

    toolRule = ' | '.join(tool_names)
    all_rules = (
        starting_grammar_rules + 
        [f'toolCall ::= "TOOL " ({toolRule})'] +
        tool_rules
    )
    grammar_string = '\n'.join(all_rules)
    grammar = LlamaGrammar.from_string(grammar_string)
    return grammar