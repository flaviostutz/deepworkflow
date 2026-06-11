"""Markdown parser with broken nested code-block detection and no escape handling."""


def parse_markdown(text: str) -> dict:
    lines = text.splitlines()
    sections = {}
    current_section = None
    current_content = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue

        if not in_code_block and line.startswith("# "):
            if current_section:
                sections[current_section] = "\n".join(current_content)
            current_section = line[2:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content)

    return sections


def extract_links(text: str) -> list[tuple[str, str]]:
    import re
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    return pattern.findall(text)


def extract_code_blocks(text: str) -> list[str]:
    blocks = []
    in_block = False
    current = []
    for line in text.splitlines():
        if line.startswith("```"):
            if in_block:
                blocks.append("\n".join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)
    return blocks


def strip_markdown(text: str) -> str:
    import re
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text
