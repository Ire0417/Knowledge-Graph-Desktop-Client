from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler

from app.api import upload, extract, graph, qa, visual
from app.config import Config
from app.services.cleanup_service import cleanup_expired_storage


def _configure_logging(app: Flask) -> None:
    log_file = app.config.get('BACKEND_LOG_FILE')
    if not log_file:
        return

    handler = RotatingFileHandler(log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.propagate = False

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    _configure_logging(app)

    # 配置CORS，允许来自历史前端地址的请求。
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3004", "http://localhost:3006", "http://127.0.0.1:3004", "http://127.0.0.1:3006"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "supports_credentials": True
        }
    })

    # 注册API蓝图
    app.register_blueprint(upload.bp, url_prefix='/upload')
    app.register_blueprint(extract.bp, url_prefix='/extract')
    app.register_blueprint(graph.bp, url_prefix='/graph')
    app.register_blueprint(visual.bp, url_prefix='/visual')
    app.register_blueprint(qa.bp, url_prefix='/qa')

    # 健康检查
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'ok', 'logFile': app.config.get('BACKEND_LOG_FILE', '')})

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(_err):
        app.logger.warning('upload rejected: request entity too large')
        return jsonify({'success': False, 'message': '文件过大，最大限制 100MB'}), 413

    @app.errorhandler(Exception)
    def handle_uncaught(err):
        if isinstance(err, HTTPException):
            app.logger.info('http error: %s %s', err.code, err.description)
            return jsonify({'success': False, 'message': err.description}), err.code

        app.logger.exception('uncaught server error: %s', str(err))
        return jsonify({'success': False, 'message': f'服务异常: {str(err)}'}), 500

    return app


app = create_app()


def _run_cleanup_once() -> None:
    report = cleanup_expired_storage(
        upload_folder=app.config.get('UPLOAD_FOLDER', ''),
        vector_db_path=app.config.get('VECTOR_DB_PATH', ''),
        expire_days=app.config.get('AUTO_CLEANUP_EXPIRE_DAYS', 3),
    )
    removed_ids = report.get('removed_file_ids', [])
    if removed_ids:
        upload.remove_file_records(removed_ids)

    removed_upload_files = report.get('removed_upload_files', 0)
    removed_vector_dirs = report.get('removed_vector_dirs', 0)
    if removed_upload_files or removed_vector_dirs:
        app.logger.info(
            f"[cleanup] removed uploads={removed_upload_files}, "
            f"vectors={removed_vector_dirs}, expire_days={report.get('expire_days', 3)}"
        )


def start_cleanup_worker() -> None:
    if not app.config.get('AUTO_CLEANUP_ENABLED', True):
        return

    interval_hours = max(1, int(app.config.get('AUTO_CLEANUP_INTERVAL_HOURS', 6)))
    interval_seconds = interval_hours * 60 * 60

    def _worker() -> None:
        while True:
            try:
                _run_cleanup_once()
            except Exception as e:
                app.logger.exception("[cleanup] failed: %s", str(e))
            time.sleep(interval_seconds)

    # 先执行一次，确保重启服务后立即清理一次过期数据。
    _run_cleanup_once()
    threading.Thread(target=_worker, daemon=True, name='cleanup-worker').start()

def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = True) -> None:
    # Flask debug reloader 会启动两个进程，仅在实际服务进程中启动清理线程。
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not debug:
        start_cleanup_worker()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)