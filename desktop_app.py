import importlib.util
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from werkzeug.serving import make_server


SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".log", ".md", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


class BackendServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    @staticmethod
    def _project_root() -> Path:
        if getattr(sys, "frozen", False):
            return Path(getattr(sys, "_MEIPASS"))
        return Path(__file__).resolve().parent

    def start(self) -> None:
        backend_root = self._project_root() / "backend"
        app_file = backend_root / "app.py"
        if not app_file.exists():
            raise RuntimeError(f"未找到后端入口: {app_file}")

        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))

        spec = importlib.util.spec_from_file_location("desktop_backend_entry", app_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 backend/app.py")

        backend_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend_module)

        backend_module.start_cleanup_worker()
        self._server = make_server(self.host, self.port, backend_module.app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, self._url(path), timeout=180, **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 413:
                raise RuntimeError("文件过大，最大限制 100MB") from exc
            message = f"HTTP {response.status_code}"
            try:
                payload = response.json()
                message = payload.get("message", message)
            except ValueError:
                text = (response.text or "").strip()
                if text:
                    message = text[:240]
            raise RuntimeError(message) from exc
        payload = response.json()
        if not payload.get("success", False) and "health" not in payload:
            raise RuntimeError(payload.get("message", "请求失败"))
        return payload

    def health(self) -> dict[str, Any]:
        response = self.session.get(self._url("/health"), timeout=10)
        response.raise_for_status()
        return response.json()

    def upload_file(self, file_path: str) -> dict[str, Any]:
        with open(file_path, "rb") as file_obj:
            files = {"file": (os.path.basename(file_path), file_obj)}
            return self._request("POST", "/upload", files=files)

    def list_files(self) -> list[dict[str, Any]]:
        return self._request("GET", "/upload/files").get("files", [])

    def parse_file(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/upload/parse", json={"fileId": file_id})

    def delete_file(self, file_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/upload/files/{file_id}")

    def extract(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/extract/", json={"fileId": file_id})

    def build_graph(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/graph/build", json={"fileId": file_id})

    def align_entities(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/graph/align", json={"fileId": file_id})

    def merge_relations(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/graph/merge", json={"fileId": file_id})

    def optimize_graph(self, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/graph/optimize", json={"fileId": file_id})

    def graph_data(self, file_id: str) -> dict[str, Any]:
        return self._request("GET", f"/graph/data/{file_id}").get("data", {})

    def qa_health(self, file_id: str | None = None) -> dict[str, Any]:
        params = {"fileId": file_id} if file_id else None
        return self._request("GET", "/qa/health", params=params)

    def ask(self, question: str, file_id: str) -> dict[str, Any]:
        return self._request("POST", "/qa/ask", json={"question": question, "fileId": file_id})

    def related(self, question: str, file_id: str) -> list[str]:
        return self._request("POST", "/qa/related", json={"question": question, "fileId": file_id}).get("questions", [])

    def history(self, file_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/qa/history/{file_id}").get("history", [])

    def clear_history(self, file_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/qa/history/{file_id}")


class MainWindow(QMainWindow):
    upload_progress_changed = Signal(int, str)

    def __init__(self, api: ApiClient, backend_server: BackendServer) -> None:
        super().__init__()
        self.api = api
        self.backend_server = backend_server
        self.files: list[dict[str, Any]] = []
        self._threads: list[QThread] = []
        self._upload_anim_phase = 0
        self._upload_running = False
        self._view_mode = "list"
        self.file_view_stack: QStackedWidget | None = None

        self.upload_progress_changed.connect(self._on_upload_progress_changed)
        self._upload_anim_timer = QTimer(self)
        self._upload_anim_timer.setInterval(140)
        self._upload_anim_timer.timeout.connect(self._tick_upload_bar)

        self.setWindowTitle("知识图谱桌面系统")
        self.resize(1280, 820)
        self._apply_styles()

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        self.nav = QListWidget()
        self.nav.setFixedWidth(210)
        for name in ["首页", "文件上传", "图谱构建", "智能问答"]:
            item = QListWidgetItem(name)
            self.nav.addItem(item)
        self.nav.setCurrentRow(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_home_page())
        self.stack.addWidget(self._build_upload_page())
        self.stack.addWidget(self._build_graph_page())
        self.stack.addWidget(self._build_qa_page())

        content_wrap = QWidget()
        content_layout = QVBoxLayout(content_wrap)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_top_bar())
        content_layout.addWidget(self.stack, 1)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        root_layout.addWidget(self.nav)
        root_layout.addWidget(content_wrap, 1)
        self.setCentralWidget(root)

        self.refresh_files()

    def _apply_styles(self) -> None:
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f6f7fa, stop:0.5 #eef1f5, stop:1 #e7ebf0);
                color: #1b2b3a;
            }
            QListWidget {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(255, 255, 255, 0.9);
                border-radius: 18px;
                color: #304152;
                padding: 10px;
                outline: 0;
            }
            QListWidget::item {
                border-radius: 12px;
                padding: 12px 14px;
                margin: 4px 0;
            }
            QListWidget::item:selected {
                background: rgba(13, 34, 61, 0.94);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.7);
            }
            QStackedWidget {
                background: transparent;
            }
            QFrame#topBar {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(255, 255, 255, 0.95);
                border-radius: 16px;
            }
            QFrame#statCard {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.95);
                border-radius: 16px;
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.58);
                border: 1px solid rgba(255, 255, 255, 0.9);
                border-radius: 12px;
                padding: 8px 12px;
                color: #2a3a48;
            }
            QLineEdit:focus {
                border: 1px solid rgba(75, 103, 132, 0.62);
            }
            QGroupBox {
                border: 1px solid rgba(255, 255, 255, 0.96);
                border-radius: 16px;
                margin-top: 12px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.78);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 2px 6px;
                color: #2f4355;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.54), stop:1 rgba(233, 240, 248, 0.34));
                border: 1px solid rgba(255, 255, 255, 0.95);
                border-radius: 14px;
                color: #1f3345;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.72), stop:1 rgba(243, 249, 255, 0.45));
                border: 1px solid rgba(255, 255, 255, 1.0);
            }
            QPushButton:pressed {
                background: rgba(211, 223, 236, 0.6);
                border: 1px solid rgba(143, 168, 192, 0.85);
            }
            QPushButton:disabled {
                background: rgba(234, 238, 244, 0.45);
                border: 1px solid rgba(220, 227, 235, 0.7);
                color: rgba(54, 70, 88, 0.45);
            }
            QPushButton#ghostButton {
                background: rgba(255, 255, 255, 0.38);
            }
            QTableWidget, QTextEdit, QPlainTextEdit, QComboBox {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                color: #213344;
                selection-background-color: rgba(183, 205, 225, 0.9);
            }
            QHeaderView::section {
                background: rgba(227, 235, 243, 0.95);
                color: #284056;
                border: 0;
                padding: 8px;
            }
            QLabel#heroTitle {
                font-size: 30px;
                font-weight: 800;
                color: #1f3449;
            }
            QLabel#heroSubTitle {
                font-size: 14px;
                color: rgba(36, 56, 75, 0.72);
            }
            QLabel#muted {
                color: rgba(41, 60, 78, 0.65);
            }
            QProgressBar#uploadProgressBar {
                border: 1px solid rgba(177, 195, 212, 0.9);
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.86);
                text-align: center;
                color: #153247;
                font-weight: 700;
            }
            QProgressBar#uploadProgressBar::chunk {
                border-radius: 10px;
                margin: 1px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d2a49, stop:0.5 #1a4c74, stop:1 #0d2a49);
            }
            """
        )

    def _build_top_bar(self) -> QWidget:
        top = QFrame()
        top.setObjectName("topBar")
        layout = QHBoxLayout(top)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        search_box = QLineEdit()
        search_box.setPlaceholderText("Search")
        search_box.setClearButtonEnabled(True)
        search_box.setMinimumWidth(320)

        date_label = QLabel(datetime.now().strftime("%Y-%m-%d"))
        date_label.setObjectName("muted")

        self.card_btn = QPushButton("Card")
        self.list_btn = QPushButton("List")
        self.card_btn.clicked.connect(lambda: self._set_view_mode("card"))
        self.list_btn.clicked.connect(lambda: self._set_view_mode("list"))
        self._apply_view_button_state()

        layout.addWidget(search_box, 1)
        layout.addWidget(date_label)
        layout.addStretch(1)
        layout.addWidget(self.card_btn)
        layout.addWidget(self.list_btn)
        return top

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        title = QLabel("知识工作台")
        title.setObjectName("heroTitle")
        desc = QLabel("桌面流程：上传文件 -> 解析抽取 -> 构建图谱 -> 智能问答")
        desc.setObjectName("heroSubTitle")

        stats_panel = QFrame()
        stats_panel.setObjectName("statCard")
        stats_layout = QHBoxLayout(stats_panel)
        stats_layout.setContentsMargins(16, 12, 16, 12)
        stats_layout.setSpacing(16)

        stats_group = QGroupBox("系统状态")
        stats_layout = QGridLayout(stats_group)
        self.backend_status = QLabel("后端状态: 检测中...")
        self.file_count = QLabel("文件数量: 0")
        self.current_file = QLabel("当前文件: 无")
        stats_layout.addWidget(self.backend_status, 0, 0)
        stats_layout.addWidget(self.file_count, 0, 1)
        stats_layout.addWidget(self.current_file, 1, 0, 1, 2)

        tasks_group = QGroupBox("任务节奏")
        tasks_layout = QVBoxLayout(tasks_group)
        tips = QLabel("建议顺序：先批量上传，再集中抽取，最后进行问答验证。")
        tips.setWordWrap(True)
        tips.setObjectName("muted")
        tasks_layout.addWidget(tips)

        stats_panel.layout().addWidget(stats_group, 2)
        stats_panel.layout().addWidget(tasks_group, 1)

        refresh_btn = QPushButton("刷新状态")
        refresh_btn.clicked.connect(self.refresh_all)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(stats_panel)
        layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return page

    def _build_upload_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        actions_group = QGroupBox("文件操作")
        actions = QHBoxLayout(actions_group)
        self.upload_btn = QPushButton("选择并上传文件")
        self.upload_btn.clicked.connect(self.upload_files)
        self.parse_btn = QPushButton("解析选中文件")
        self.parse_btn.clicked.connect(self.parse_selected_file)
        self.delete_btn = QPushButton("删除选中文件")
        self.delete_btn.clicked.connect(self.delete_selected_file)
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh_files)
        actions.addWidget(self.upload_btn)
        actions.addWidget(self.parse_btn)
        actions.addWidget(self.delete_btn)
        actions.addWidget(self.refresh_btn)
        actions.addStretch(1)

        self.upload_progress_label = QLabel("待上传")
        self.upload_progress_label.setObjectName("muted")
        self.upload_progress_bar = QProgressBar()
        self.upload_progress_bar.setObjectName("uploadProgressBar")
        self.upload_progress_bar.setMinimum(0)
        self.upload_progress_bar.setMaximum(100)
        self.upload_progress_bar.setValue(0)
        self.upload_progress_bar.setFormat("0%")
        self.upload_progress_bar.setTextVisible(True)
        self.upload_progress_bar.setFixedHeight(24)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.upload_progress_label)
        progress_layout.addWidget(self.upload_progress_bar, 1)

        table_group = QGroupBox("文件列表")
        table_layout = QVBoxLayout(table_group)
        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["文件ID", "名称", "状态", "上传时间"])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.file_table)

        card_group = QGroupBox("文件卡片")
        card_layout = QVBoxLayout(card_group)
        self.file_card_list = QListWidget()
        self.file_card_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        card_layout.addWidget(self.file_card_list)

        self.file_view_stack = QStackedWidget()
        self.file_view_stack.addWidget(table_group)
        self.file_view_stack.addWidget(card_group)
        self.file_view_stack.setCurrentIndex(0 if self._view_mode == "list" else 1)

        log_group = QGroupBox("上传与解析日志")
        log_layout = QVBoxLayout(log_group)
        self.upload_log = QPlainTextEdit()
        self.upload_log.setReadOnly(True)
        self.upload_log.setPlaceholderText("这里会显示上传和解析日志")
        log_layout.addWidget(self.upload_log)

        layout.addWidget(actions_group)
        layout.addLayout(progress_layout)
        layout.addWidget(self.file_view_stack, 3)
        layout.addWidget(log_group, 2)
        return page

    def _set_upload_controls_enabled(self, enabled: bool) -> None:
        self.upload_btn.setEnabled(enabled)
        self.parse_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)

    def _set_view_mode(self, mode: str) -> None:
        if mode == self._view_mode:
            return
        self._view_mode = mode
        if self.file_view_stack is not None:
            self.file_view_stack.setCurrentIndex(0 if mode == "list" else 1)
        self._apply_view_button_state()

    def _apply_view_button_state(self) -> None:
        if self._view_mode == "list":
            self.list_btn.setEnabled(False)
            self.card_btn.setEnabled(True)
            self.list_btn.setObjectName("")
            self.card_btn.setObjectName("ghostButton")
        else:
            self.list_btn.setEnabled(True)
            self.card_btn.setEnabled(False)
            self.list_btn.setObjectName("ghostButton")
            self.card_btn.setObjectName("")

        for btn in (self.list_btn, self.card_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()

    def _upload_chunk_style(self, phase: int) -> str:
        offset = (phase % 10) / 10.0
        s1 = max(0.0, offset - 0.18)
        s2 = min(1.0, offset + 0.12)
        s3 = min(1.0, offset + 0.42)
        return (
            "QProgressBar#uploadProgressBar::chunk {"
            "border-radius: 10px;"
            "margin: 1px;"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"stop:0 #0d2a49, stop:{s1:.2f} #0d2a49, stop:{s2:.2f} #1a4c74, "
            f"stop:{s3:.2f} #0f3556, stop:1 #1a4c74);"
            "}"
        )

    def _apply_upload_chunk_style(self) -> None:
        self.upload_progress_bar.setStyleSheet(self._upload_chunk_style(self._upload_anim_phase))

    def _tick_upload_bar(self) -> None:
        if not self._upload_running:
            return
        self._upload_anim_phase += 1
        self._apply_upload_chunk_style()

    def _set_upload_running(self, running: bool) -> None:
        self._upload_running = running
        self._set_upload_controls_enabled(not running)
        if running:
            self._upload_anim_phase = 0
            self._apply_upload_chunk_style()
            self._upload_anim_timer.start()
        else:
            self._upload_anim_timer.stop()

    def _on_upload_progress_changed(self, percent: int, status_text: str) -> None:
        percent = max(0, min(100, percent))
        self.upload_progress_bar.setValue(percent)
        self.upload_progress_bar.setFormat(f"{percent}%")
        self.upload_progress_label.setText(status_text)

    def _build_graph_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        top_group = QGroupBox("图谱目标")
        top = QHBoxLayout(top_group)
        self.graph_file_combo = QComboBox()
        self.graph_file_combo.setMinimumWidth(420)
        top.addWidget(QLabel("目标文件:"))
        top.addWidget(self.graph_file_combo)
        top.addStretch(1)

        actions_group = QGroupBox("图谱动作")
        buttons = QHBoxLayout(actions_group)
        for text, handler in [
            ("执行抽取", self.extract_file),
            ("构建图谱", self.build_graph),
            ("实体对齐", self.align_entities),
            ("关系合并", self.merge_relations),
            ("图谱优化", self.optimize_graph),
            ("刷新图谱数据", self.refresh_graph_data),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            buttons.addWidget(btn)

        data_group = QGroupBox("图谱结果")
        data_layout = QVBoxLayout(data_group)
        self.graph_stats = QLabel("图谱统计: 暂无")
        self.graph_data_view = QPlainTextEdit()
        self.graph_data_view.setReadOnly(True)
        data_layout.addWidget(self.graph_stats)
        data_layout.addWidget(self.graph_data_view, 1)

        layout.addWidget(top_group)
        layout.addWidget(actions_group)
        layout.addWidget(data_group, 1)
        return page

    def _build_qa_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        file_group = QGroupBox("问答目标")
        row = QHBoxLayout(file_group)
        self.qa_file_combo = QComboBox()
        self.qa_file_combo.setMinimumWidth(420)
        health_btn = QPushButton("检测问答服务")
        health_btn.clicked.connect(self.check_qa_health)
        row.addWidget(QLabel("问答文件:"))
        row.addWidget(self.qa_file_combo)
        row.addWidget(health_btn)
        row.addStretch(1)

        status_group = QGroupBox("服务状态")
        status_layout = QVBoxLayout(status_group)
        self.qa_health_label = QLabel("问答服务状态: 未检测")
        status_layout.addWidget(self.qa_health_label)

        history_group = QGroupBox("历史记录")
        history_layout = QVBoxLayout(history_group)
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        history_layout.addWidget(self.history_view)

        question_group = QGroupBox("提问输入")
        question_layout = QVBoxLayout(question_group)
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("输入问题，支持回车换行")
        self.question_input.setFixedHeight(90)
        question_layout.addWidget(self.question_input)

        action = QHBoxLayout()
        ask_btn = QPushButton("提问")
        ask_btn.clicked.connect(self.ask_question)
        related_btn = QPushButton("推荐相关问题")
        related_btn.clicked.connect(self.fetch_related)
        clear_btn = QPushButton("清空历史")
        clear_btn.clicked.connect(self.clear_history)
        action.addWidget(ask_btn)
        action.addWidget(related_btn)
        action.addWidget(clear_btn)
        action.addStretch(1)
        question_layout.addLayout(action)

        answer_group = QGroupBox("回答")
        answer_layout = QVBoxLayout(answer_group)
        self.answer_view = QPlainTextEdit()
        self.answer_view.setReadOnly(True)
        self.answer_view.setPlaceholderText("回答将显示在这里")
        answer_layout.addWidget(self.answer_view)

        layout.addWidget(file_group)
        layout.addWidget(status_group)
        layout.addWidget(history_group, 2)
        layout.addWidget(question_group)
        layout.addWidget(answer_group, 2)
        return page

    def _selected_table_file_id(self) -> str | None:
        if self._view_mode == "card":
            item = self.file_card_list.currentItem()
            return item.data(Qt.ItemDataRole.UserRole) if item else None

        row = self.file_table.currentRow()
        if row < 0:
            return None
        return self.file_table.item(row, 0).text()

    def _ensure_readable_file(self, path: str) -> None:
        try:
            with open(path, "rb") as file_obj:
                file_obj.read(1)
        except PermissionError as exc:
            raise RuntimeError("文件被占用或无权限读取，请关闭文档后重试") from exc
        except OSError as exc:
            raise RuntimeError(f"无法读取文件: {exc}") from exc

    def _current_graph_file_id(self) -> str | None:
        data = self.graph_file_combo.currentData()
        return str(data) if data else None

    def _current_qa_file_id(self) -> str | None:
        data = self.qa_file_combo.currentData()
        return str(data) if data else None

    def _run_async(self, func: Callable, on_done: Callable[[Any], None], on_error: Callable[[str], None] | None = None) -> None:
        thread = QThread(self)
        worker = Worker(func)
        worker.moveToThread(thread)

        def done(result: Any) -> None:
            on_done(result)
            thread.quit()

        def fail(message: str) -> None:
            if on_error:
                on_error(message)
            else:
                self.error(message)
            thread.quit()

        worker.finished.connect(done)
        worker.failed.connect(fail)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)

        self._threads.append(thread)

        def cleanup() -> None:
            if thread in self._threads:
                self._threads.remove(thread)

        thread.finished.connect(cleanup)
        thread.start()

    def info(self, message: str) -> None:
        self.upload_log.appendPlainText(message)

    def error(self, message: str) -> None:
        QMessageBox.critical(self, "操作失败", message)
        self.info(f"[错误] {message}")

    def refresh_all(self) -> None:
        self.refresh_files()
        self._run_async(self.api.health, self._on_health_loaded)

    def _on_health_loaded(self, payload: dict[str, Any]) -> None:
        status = payload.get("status", "unknown")
        self.backend_status.setText(f"后端状态: {status}")
        log_file = payload.get("logFile")
        if log_file:
            self.info(f"后台日志文件: {log_file}")

    def refresh_files(self) -> None:
        self._run_async(self.api.list_files, self._on_files_loaded)

    def _on_files_loaded(self, files: list[dict[str, Any]]) -> None:
        self.files = files
        self.file_table.setRowCount(len(files))
        self.file_card_list.clear()
        self.graph_file_combo.clear()
        self.qa_file_combo.clear()

        for row, file_info in enumerate(files):
            values = [
                file_info.get("id", ""),
                file_info.get("name", ""),
                file_info.get("status", ""),
                file_info.get("uploadTime", ""),
            ]
            for col, value in enumerate(values):
                self.file_table.setItem(row, col, QTableWidgetItem(str(value)))

            card_text = (
                f"{file_info.get('name', '')}\n"
                f"状态: {file_info.get('status', '')} | 上传: {file_info.get('uploadTime', '')}\n"
                f"ID: {file_info.get('id', '')}"
            )
            card_item = QListWidgetItem(card_text)
            card_item.setData(Qt.ItemDataRole.UserRole, file_info.get("id", ""))
            self.file_card_list.addItem(card_item)

            display_name = f"{file_info.get('name', '')} [{file_info.get('status', '')}]"
            file_id = file_info.get("id", "")
            self.graph_file_combo.addItem(display_name, file_id)
            self.qa_file_combo.addItem(display_name, file_id)

        count = len(files)
        self.file_count.setText(f"文件数量: {count}")
        current_name = files[0].get("name") if files else "无"
        self.current_file.setText(f"当前文件: {current_name}")

    def upload_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要上传的文件",
            "",
            "文档文件 (*.pdf *.docx *.txt *.log *.md *.xlsx *.xls *.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not paths:
            return

        self.info(f"准备上传 {len(paths)} 个文件")
        self.upload_progress_changed.emit(0, f"正在准备上传，共 {len(paths)} 个文件")
        self._set_upload_running(True)

        sizes = {path: max(1, os.path.getsize(path)) for path in paths}
        total_size = sum(sizes.values())

        def task() -> dict[str, list[str]]:
            done_size = 0
            success_messages: list[str] = []
            error_messages: list[str] = []

            for index, path in enumerate(paths, 1):
                name = os.path.basename(path)
                start_percent = int(done_size * 100 / total_size)
                self.upload_progress_changed.emit(start_percent, f"上传中 {index}/{len(paths)}: {name}")

                try:
                    lower_name = name.lower()
                    suffix = Path(name).suffix.lower()
                    if lower_name.startswith("~$"):
                        raise RuntimeError("检测到 Office 临时锁文件（~$ 开头），请关闭文档后上传原始 .docx 文件")
                    if suffix == ".doc":
                        raise RuntimeError("不支持 .doc，请另存为 .docx 后重试")
                    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
                        raise RuntimeError(f"不支持的文件类型: {suffix or '无后缀'}")
                    if sizes.get(path, 0) > 100 * 1024 * 1024:
                        raise RuntimeError("文件过大，最大限制 100MB")

                    self._ensure_readable_file(path)

                    payload = self.api.upload_file(path)
                    success_messages.append(f"上传成功: {name} -> {payload.get('fileId', '')}")
                except Exception as exc:
                    error_messages.append(f"上传失败: {name} ({str(exc)})")

                done_size += sizes[path]
                finish_percent = int(done_size * 100 / total_size)
                self.upload_progress_changed.emit(finish_percent, f"已完成 {index}/{len(paths)}: {name}")

            return {"success": success_messages, "errors": error_messages}

        self._run_async(task, self._on_upload_done, self._on_upload_failed)

    def _on_upload_done(self, payload: dict[str, list[str]]) -> None:
        self._set_upload_running(False)
        self.upload_progress_changed.emit(100, "上传任务完成")

        for msg in payload.get("success", []):
            self.info(msg)

        errors = payload.get("errors", [])
        for msg in errors:
            self.info(f"[错误] {msg}")

        if errors and not payload.get("success"):
            self.error("所有文件上传失败，请查看日志中的具体错误。")
        elif errors:
            self.info(f"有 {len(errors)} 个文件上传失败，请检查日志。")

        self.refresh_files()

    def _on_upload_failed(self, message: str) -> None:
        self._set_upload_running(False)
        self.upload_progress_changed.emit(0, "上传中断")
        self.error(message)

    def parse_selected_file(self) -> None:
        file_id = self._selected_table_file_id()
        if not file_id:
            self.error("请先在文件表格中选择一行")
            return

        self.info(f"开始解析文件: {file_id}")
        self._run_async(lambda: self.api.parse_file(file_id), lambda _: self._after_simple_action("解析完成"))

    def delete_selected_file(self) -> None:
        file_id = self._selected_table_file_id()
        if not file_id:
            self.error("请先在文件表格中选择一行")
            return

        self._run_async(lambda: self.api.delete_file(file_id), lambda _: self._after_simple_action("删除完成"))

    def extract_file(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.extract(file_id), lambda _: self._after_simple_action("抽取完成"))

    def build_graph(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.build_graph(file_id), lambda _: self._after_simple_action("图谱构建完成"))

    def align_entities(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.align_entities(file_id), lambda _: self._after_simple_action("实体对齐完成"))

    def merge_relations(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.merge_relations(file_id), lambda _: self._after_simple_action("关系合并完成"))

    def optimize_graph(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.optimize_graph(file_id), lambda _: self._after_simple_action("图谱优化完成"))

    def refresh_graph_data(self) -> None:
        file_id = self._current_graph_file_id()
        if not file_id:
            self.error("请先在图谱页面选择文件")
            return

        self._run_async(lambda: self.api.graph_data(file_id), self._on_graph_data)

    def _on_graph_data(self, data: dict[str, Any]) -> None:
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        self.graph_stats.setText(f"图谱统计: 节点 {len(nodes)} | 关系 {len(links)}")
        self.graph_data_view.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))

    def check_qa_health(self) -> None:
        file_id = self._current_qa_file_id()
        self._run_async(lambda: self.api.qa_health(file_id), self._on_qa_health)

    def _on_qa_health(self, payload: dict[str, Any]) -> None:
        health = payload.get("health", {})
        status = "可用" if health.get("ok") else "异常"
        model = health.get("model", "-")
        self.qa_health_label.setText(f"问答服务状态: {status} | 模型: {model}")

    def ask_question(self) -> None:
        file_id = self._current_qa_file_id()
        question = self.question_input.toPlainText().strip()
        if not file_id:
            self.error("请选择问答文件")
            return
        if not question:
            self.error("请输入问题")
            return

        self._run_async(lambda: self.api.ask(question, file_id), self._on_answer)

    def _on_answer(self, payload: dict[str, Any]) -> None:
        self.answer_view.setPlainText(payload.get("answer", ""))
        self.load_history()

    def fetch_related(self) -> None:
        file_id = self._current_qa_file_id()
        question = self.question_input.toPlainText().strip()
        if not file_id or not question:
            self.error("请先选择文件并输入问题")
            return

        self._run_async(lambda: self.api.related(question, file_id), self._on_related)

    def _on_related(self, items: list[str]) -> None:
        if not items:
            self.answer_view.setPlainText("未返回相关问题")
            return
        self.answer_view.setPlainText("\n".join(f"- {item}" for item in items))

    def load_history(self) -> None:
        file_id = self._current_qa_file_id()
        if not file_id:
            return
        self._run_async(lambda: self.api.history(file_id), self._on_history)

    def _on_history(self, items: list[dict[str, Any]]) -> None:
        if not items:
            self.history_view.setPlainText("暂无历史")
            return

        lines = []
        for entry in items:
            lines.append(f"[{entry.get('timestamp', '')}]\n问: {entry.get('question', '')}\n答: {entry.get('answer', '')}\n")
        self.history_view.setPlainText("\n".join(lines))

    def clear_history(self) -> None:
        file_id = self._current_qa_file_id()
        if not file_id:
            self.error("请选择问答文件")
            return

        self._run_async(lambda: self.api.clear_history(file_id), lambda _: self._after_simple_action("历史已清空"))

    def _after_simple_action(self, message: str) -> None:
        self.info(message)
        self.refresh_files()
        self.load_history()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.backend_server.stop()
        super().closeEvent(event)


def main() -> None:
    backend = BackendServer()
    backend.start()

    app = QApplication(sys.argv)
    window = MainWindow(ApiClient(), backend)
    window.show()
    window.refresh_all()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
