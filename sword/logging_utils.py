"""Filters for repetitive quantization notices, without hiding other failures."""
import logging
import re
import sys
import warnings
from functools import wraps


_CAST_NOTICE = r"MatMul8bitLt: inputs will be cast from .* to (?:torch\.)?float16 during quantization\.?"


def _is_cast_notice(message):
    return re.fullmatch(_CAST_NOTICE, str(message).strip()) is not None


class _QuantizationCastFilter(logging.Filter):
    _sword_quantization_cast_filter = True

    def filter(self, record):
        return record.levelno >= logging.ERROR or not _is_cast_notice(record.getMessage())


def quiet_quantization_cast_notices():
    """Reinstall after dependency imports and notebook logging/warning resets.

    Guard the emitting logger as well as its filters: integrations can replace
    filters/handlers after Sword is imported. Older builds use warnings.warn.
    """
    module = sys.modules.get("bitsandbytes.autograd._functions")
    loggers = [logging.getLogger("bitsandbytes.autograd._functions")]
    if module is not None and isinstance(getattr(module, "logger", None), logging.Logger):
        loggers.append(module.logger)
    for logger in loggers:
        if not any(getattr(item, "_sword_quantization_cast_filter", False) for item in logger.filters):
            logger.addFilter(_QuantizationCastFilter())
        if not getattr(logger.warning, "_sword_quantization_cast_guard", False):
            original_warning = logger.warning
            @wraps(original_warning)
            def guarded_warning(message, *args, _original=original_warning, **kwargs):
                try:
                    rendered = str(message) % args if args else str(message)
                except (TypeError, ValueError):
                    rendered = str(message)
                if not _is_cast_notice(rendered):
                    return _original(message, *args, **kwargs)
            guarded_warning._sword_quantization_cast_guard = True
            logger.warning = guarded_warning
    warnings.filterwarnings("ignore", message=_CAST_NOTICE, category=UserWarning)
    if not getattr(warnings.showwarning, "_sword_quantization_cast_guard", False):
        original_showwarning = warnings.showwarning
        @wraps(original_showwarning)
        def guarded_showwarning(message, category, *args, **kwargs):
            if not (issubclass(category, UserWarning) and _is_cast_notice(message)):
                return original_showwarning(message, category, *args, **kwargs)
        guarded_showwarning._sword_quantization_cast_guard = True
        warnings.showwarning = guarded_showwarning


def output_preview(text, max_chars=1200):
    """Show the beginning and end so thinking cannot hide the final answer."""
    text = str(text or "")
    if not text:
        return "<empty>"
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    return text[:head] + f"\n... [{len(text) - max_chars} characters omitted] ...\n" + text[-(max_chars - head):]
