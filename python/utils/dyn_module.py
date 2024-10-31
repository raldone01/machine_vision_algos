import importlib.util
import sys
import os
from importlib import reload
from pathlib import Path

from utils.benchmarking import LogTimer


def get_module_name_from_file_path(file_path: str):
    return Path(file_path).stem


def get_module(file_path: str):
    module_name = get_module_name_from_file_path(file_path)
    return sys.modules.get(module_name)


def load_module(file_path: str):
    module_name = get_module_name_from_file_path(file_path)

    already_loaded = module_name in sys.modules
    if already_loaded:
        with LogTimer(f"Reloading {module_name}"):
            module = sys.modules[module_name]
            reload(module)
        assert module.__name__ == module_name
        return module

    with LogTimer(f"Loading {module_name}"):
        folder = str(Path(file_path).parent)
        if folder not in sys.path:
            sys.path.append(folder)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert module.__name__ == module_name

    return module


def load_modules(folder_path: str) -> list[str]:
    module_file_paths = [f for f in os.listdir(folder_path) if str(f).endswith(".py")]
    module_file_paths.sort()

    module_names = []

    with LogTimer(f"Loading {len(module_file_paths)} modules"):
        for image_filename in module_file_paths:
            full_path = Path(folder_path) / image_filename
            module = load_module(full_path)
            module_names.append(module.__name__)

    return module_names
