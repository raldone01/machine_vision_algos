import re
import inspect


def get_call_site_source_code(current_frame) -> str:
    """
    current_frame = inspect.currentframe()
    """
    # collect caller information
    caller = inspect.getframeinfo(current_frame.f_back, context=0xFFFFFFFFFFFFFFFF)
    positions = caller.positions

    function_call = caller.code_context[positions.lineno - 1 : positions.end_lineno]

    # Remove whitespaces and newlines
    call_site_source_code = re.sub("\s+", " ", "".join(function_call))
    return call_site_source_code
