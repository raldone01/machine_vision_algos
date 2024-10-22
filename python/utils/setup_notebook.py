import re


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def source_code_path_is_from_notebook(source_file: str) -> bool:
    if not is_notebook():
        return False
    # /tmp/ipykernel_30141/770546325.py
    # match <anypath>/ipykernel_<pid>/<hash>.py
    return re.match(r".*/ipykernel_\d+/\d+\.py", source_file) is not None


def init_notebook() -> None:
    try:
        if is_notebook():
            import colorama.ansitowin32 as ansito_win32

            # https://github.com/tartley/colorama/blob/136808718af8b9583cb2eed1756ed6972eda4975/colorama/ansitowin32.py#L49
            # monkey patching colorama to work with jupyter
            # replace StreamWrapper.isatty with a lambda that returns True
            ansito_win32.StreamWrapper.isatty = lambda self: True
    except ImportError:
        pass

    try:
        import matplotlib.pyplot as plt
        import matplotlib

        # https://matplotlib.org/stable/users/explain/figure/backends.html#interactive-backends
        matplotlib.use("ipympl")
        # matplotlib.use("TkAgg")
        # matplotlib.use("QtAgg")

        plt.style.use("ggplot")
        plt.style.use("fast")
    except ImportError:
        pass
