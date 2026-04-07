from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from flask import Flask, render_template
from jinja2 import ChoiceLoader, FileSystemLoader


BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent / "backend"


def _load_backend_app():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

    backend_app_file = BACKEND_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("backend_app", backend_app_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载后端入口: {backend_app_file}")
    backend_module = importlib.util.module_from_spec(spec)
    sys.modules["backend_app"] = backend_module
    spec.loader.exec_module(backend_module)
    backend_app: Flask = backend_module.app

    template_dir = BASE_DIR / "templates"
    backend_app.jinja_loader = ChoiceLoader(
        [backend_app.jinja_loader, FileSystemLoader(str(template_dir))]
    )

    @backend_app.get("/")
    def index():
        return render_template("index.html")

    return backend_app, backend_module


def run_flask(host: str = "127.0.0.1", port: int = 9527) -> None:
    backend_app, backend_module = _load_backend_app()
    if hasattr(backend_module, "start_cleanup_worker"):
        backend_module.start_cleanup_worker()

    backend_app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_flask()
