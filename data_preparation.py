import json

def format_prompt(prompt_str):
    messages = json.loads(prompt_str)
    formatted = ""
    for turn in messages:
        formatted += f"{turn['role']}: {turn['content']}\n"
    return formatted
