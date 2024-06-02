def construct_system_message(msg: str):
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": msg}
        ]
    }