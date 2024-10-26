import re
import os


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

    # Normalize path to ensure compatibility
    normalized_path = os.path.normpath(source_file)

    # /tmp/ipykernel_30141/770546325.py
    # C:\\Users\\_\\AppData\\Local\\Temp\\ipykernel_6700\\169566949.py
    # Matches paths like /tmp/ipykernel_<pid>/<hash>.py or C:\\Users\\...\\ipykernel_<pid>\\<hash>.py
    return re.match(r".*[/\\]ipykernel_\d+[/\\]\d+\.py", normalized_path) is not None


notebook_dir = None


def get_notebook_dir() -> str:
    return notebook_dir


def init_notebook() -> None:
    global notebook_dir
    if is_notebook():
        notebook_dir = os.getcwd()

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
