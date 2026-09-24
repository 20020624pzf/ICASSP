"""Load archived helpers in a path-specific namespace without changing sys.path.

In particular, an admitted worker's existing common/pll/audio_core modules
must neither be reused nor replaced. Loading this module loads no model.
"""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import threading

VENDOR = Path(__file__).resolve().parent / "vendor"
NAMESPACE = "_asr_reproduction_" + hashlib.sha256(str(VENDOR).encode()).hexdigest()[:24]
_LOCK = threading.RLock()


def load_vendor(name):
    if name not in {"common", "ir_common", "audio_core", "pll", "risk_selector",
                    "wer_rules_v3", "cluster_bootstrap"}:
        raise ValueError(f"Unknown archived helper: {name}")
    with _LOCK:
        if NAMESPACE not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                NAMESPACE, VENDOR / "__init__.py",
                submodule_search_locations=[str(VENDOR)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[NAMESPACE] = package
            spec.loader.exec_module(package)
        package = sys.modules[NAMESPACE]
        if Path(package.__file__).resolve() != VENDOR / "__init__.py":
            raise RuntimeError("Reproduction namespace is bound to another package")
        return importlib.import_module("." + name, NAMESPACE)
