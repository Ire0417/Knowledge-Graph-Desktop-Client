from __future__ import annotations

import importlib
import multiprocessing
import time
import urllib.request

from PySide6.QtCore import QStandardPaths, QTimer, QUrl
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTextBrowser

from app import run_flask


HOST = "127.0.0.1"
PORT = 9527
BASE_URL = f"http://{HOST}:{PORT}/"


def _wait_for_server(url: str, timeout_seconds: int = 10) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "health", timeout=1):
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _start_backend() -> multiprocessing.Process:
    process = multiprocessing.Process(target=run_flask, daemon=True)
    process.start()
    return process


def _create_web_view(parent: QMainWindow):
    try:
        webengine = importlib.import_module("PySide6.QtWebEngineWidgets")
        web_view_class = getattr(webengine, "QWebEngineView")
        return web_view_class(parent)
    except Exception:
        fallback = QTextBrowser(parent)
        fallback.setOpenExternalLinks(True)
        return fallback


class MainWindow(QMainWindow):
    def __init__(self, backend_ready: bool) -> None:
        super().__init__()
        self.setWindowTitle("DesktopAPP")
        self.resize(1100, 720)

        self.view = _create_web_view(self)
        self.setCentralWidget(self.view)
        if backend_ready:
            if hasattr(self.view, "load"):
                self.view.load(QUrl(BASE_URL))
            else:
                self._show_error_page(
                    "Web engine not available",
                    "PyQt6-WebEngine is required to render the app UI.",
                )
        else:
            self._show_error_page(
                "Backend did not start",
                f"Could not reach {BASE_URL}api/health within timeout.",
            )

        if hasattr(self.view, "loadFinished"):
            self.view.loadFinished.connect(self._on_load_finished)

        self._setup_downloads()

    def _setup_downloads(self) -> None:
        if not hasattr(self.view, "page"):
            return
        page = self.view.page()
        if page is None:
            return
        profile = page.profile()
        if profile is None:
            return
        profile.downloadRequested.connect(self._on_download_requested)

    def _on_download_requested(self, item) -> None:
        download_dir = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
        if download_dir:
            item.setDownloadDirectory(download_dir)
        try:
            item.finished.connect(lambda: self._on_download_finished(item))
        except Exception:
            pass
        item.accept()

    def _on_download_finished(self, item) -> None:
        if hasattr(item, "path"):
            path = item.path()
        else:
            path = ""
        if not path and hasattr(item, "downloadFileName"):
            path = item.downloadFileName()
        if path:
            QMessageBox.information(self, "下载完成", f"已保存到: {path}")
        else:
            QMessageBox.information(self, "下载完成", "下载已完成。")

    def _on_load_finished(self, ok: bool) -> None:
        if ok:
            return
        self._show_error_page("Page failed to load", f"Failed to load {BASE_URL}.")

    def _show_error_page(self, title: str, detail: str) -> None:
        html = (
            "<html><body style='font-family:Segoe UI, Arial, sans-serif;'>"
            f"<h2>{title}</h2>"
            f"<p>{detail}</p>"
            "<p>Check backend logs or port availability.</p>"
            "</body></html>"
        )
        if hasattr(self.view, "load") and hasattr(self.view, "setHtml"):
            # QWebEngineView.setHtml supports a base URL.
            self.view.setHtml(html, QUrl(BASE_URL))
        elif hasattr(self.view, "setHtml"):
            self.view.setHtml(html)
        else:
            self.view.setText(html)


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    backend = _start_backend()
    backend_ready = _wait_for_server(BASE_URL)

    app = QApplication([])
    window = MainWindow(backend_ready)
    window.show()
    exit_code = app.exec()

    if backend.is_alive():
        backend.terminate()
        backend.join(timeout=2)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
