# Importable wrapper for 3_analysis_LAI.py (which can't be imported directly due to leading digit).
# The 3_analysis_LAI.py script is kept as the CLI entry point referenced by Snakemake configs.
import importlib.util
import os
from types import FunctionType as _FT

_spec = importlib.util.spec_from_file_location(
    "_analysis_lai",
    os.path.join(os.path.dirname(__file__), "3_analysis_LAI.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export all public names

for _name in dir(_mod):
    _obj = getattr(_mod, _name)
    if not _name.startswith("_") and (isinstance(_obj, _FT) or not callable(_obj)):
        globals()[_name] = _obj

del _FT, _name, _obj, _spec, _mod
