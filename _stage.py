"""Import helper: the stage files are named 01_...py so that `ls` shows the
pipeline in order, but Python module names may not start with a digit. This
shim makes `load("03_model")` work from any working directory."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load(name: str):
    """name like "03_model" -> the module, imported once and cached."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
