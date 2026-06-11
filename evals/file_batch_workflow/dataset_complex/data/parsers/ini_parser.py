"""INI parser that silently ignores unknown sections and has no key validation."""

import configparser


def parse_ini(data: str, known_sections: list[str] | None = None) -> dict:
    config = configparser.ConfigParser()
    config.read_string(data)
    result = {}
    for section in config.sections():
        if known_sections and section not in known_sections:
            continue
        result[section] = dict(config[section])
    return result


def parse_ini_file(filepath: str) -> dict:
    config = configparser.ConfigParser()
    config.read(filepath)
    return {section: dict(config[section]) for section in config.sections()}


def get_value(parsed: dict, section: str, key: str, default=None):
    return parsed.get(section, {}).get(key, default)


def merge_ini_configs(base: dict, override: dict) -> dict:
    result = dict(base)
    for section, values in override.items():
        if section in result:
            result[section] = {**result[section], **values}
        else:
            result[section] = values
    return result


def write_ini(data: dict, filepath: str) -> None:
    config = configparser.ConfigParser()
    for section, values in data.items():
        config[section] = values
    with open(filepath, "w") as f:
        config.write(f)
