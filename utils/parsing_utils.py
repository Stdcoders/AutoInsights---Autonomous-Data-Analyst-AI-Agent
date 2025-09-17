# utils/parsing_utils.py
def parse_keyvals(input_text: str) -> dict:
    """
    Parse 'key=value' style input into a dict.
    Robust: handles Action Input:, newlines, stray semicolons.
    """
    text = input_text.strip()
    if "Action Input:" in text:
        text = text.split("Action Input:")[-1].strip()

    parts = {}
    for p in text.replace("\n", ";").split(";"):
        if "=" in p:
            k, v = p.split("=", 1)
            parts[k.strip().lower()] = v.strip()

    return parts
