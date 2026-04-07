import re


def extract_code_from_text(text: str, code_block_tags: tuple[str, str]) -> str | None:
    """Extract code from the LLM's output."""
    pattern = rf"{code_block_tags[0]}(.*?){code_block_tags[1]}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None
