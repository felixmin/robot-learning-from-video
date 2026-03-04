from __future__ import annotations

import builtins
import importlib
import sys


def test_online_laq_import_does_not_require_laq_inference(monkeypatch):
    sys.modules.pop("foundation.online_laq", None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "laq.inference":
            raise ModuleNotFoundError("No module named 'laq.inference'")
        if name == "laq" and fromlist and "inference" in fromlist:
            raise ModuleNotFoundError("No module named 'laq.inference'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("foundation.online_laq")
    assert hasattr(module, "frames_to_laq_video")
    assert hasattr(module, "LAQTaskCodeProvider")
