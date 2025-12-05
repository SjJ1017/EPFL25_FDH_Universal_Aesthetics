import json

def load_multiline_jsonl(path):
    data = []
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            buffer += line
            try:
                obj = json.loads(buffer)
                data.append(obj)
                buffer = "" 
            except json.JSONDecodeError:
                continue
    if buffer.strip():
        raise ValueError("Incomplete JSON object at end of file")
    return data
