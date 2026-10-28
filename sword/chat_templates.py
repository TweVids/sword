"""Read explicit chat templates before allocating or downloading model weights."""
from pathlib import Path


class RenderedChatTokenizer:
    """Serving prompts already contain the template's BOS and role tokens."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self.tokenizer(*args, **kwargs)


def read_chat_template(path):
    if path is None:
        return None
    if not str(path).strip():
        raise ValueError("chat_template_path must name a nonempty Jinja template file")
    template_path = Path(path).expanduser().resolve()
    template = template_path.read_text(encoding="utf-8-sig")
    if not template.strip():
        raise ValueError(f"Chat template is empty: {template_path}")
    from jinja2 import Environment, TemplateSyntaxError
    try:
        Environment().parse(template)
    except TemplateSyntaxError as exc:
        raise ValueError(f"Invalid chat template {template_path}: {exc}") from exc
    return template
