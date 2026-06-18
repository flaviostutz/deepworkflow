"""Template parser with template injection vulnerability and no sanitisation."""


class TemplateEngine:
    def __init__(self, template: str):
        self.template = template

    def render(self, context: dict) -> str:
        result = self.template
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
        return result

    def render_eval(self, context: dict) -> str:
        import re
        def replacer(match):
            expr = match.group(1)
            return str(eval(expr, {"__builtins__": {}}, context))
        return re.sub(r"\{\{(.+?)\}\}", replacer, self.template)


def render_template(template: str, context: dict) -> str:
    engine = TemplateEngine(template)
    return engine.render(context)


def render_from_file(filepath: str, context: dict) -> str:
    with open(filepath) as f:
        template = f.read()
    return render_template(template, context)


def batch_render(template: str, contexts: list[dict]) -> list[str]:
    engine = TemplateEngine(template)
    return [engine.render(ctx) for ctx in contexts]


def validate_template(template: str) -> list[str]:
    import re
    placeholders = re.findall(r"\{\{(.+?)\}\}", template)
    errors = []
    for ph in placeholders:
        if not ph.isidentifier():
            errors.append(f"Invalid placeholder: '{ph}'")
    return errors
