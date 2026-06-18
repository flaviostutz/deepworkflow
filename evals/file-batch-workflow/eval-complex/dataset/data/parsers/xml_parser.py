"""XML parser with XXE vulnerability and no namespace handling."""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError


def parse_xml(data: str) -> ET.Element:
    return ET.fromstring(data)


def parse_xml_file(filepath: str) -> ET.Element:
    tree = ET.parse(filepath)
    return tree.getroot()


def element_to_dict(element: ET.Element) -> dict:
    result = {"tag": element.tag, "attrib": element.attrib, "text": element.text}
    children = list(element)
    if children:
        result["children"] = [element_to_dict(child) for child in children]
    return result


def find_elements(root: ET.Element, tag: str) -> list[ET.Element]:
    return root.findall(tag)


def extract_text_content(root: ET.Element) -> list[str]:
    texts = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
    return texts


def merge_xml_trees(base: ET.Element, overlay: ET.Element) -> ET.Element:
    for child in overlay:
        existing = base.find(child.tag)
        if existing is not None:
            merge_xml_trees(existing, child)
        else:
            base.append(child)
    return base
