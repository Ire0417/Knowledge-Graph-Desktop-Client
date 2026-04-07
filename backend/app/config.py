import os
import sys


def _resolve_data_root() -> str:
    if getattr(sys, "frozen", False):
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "ZhishiExeDesktopData")

    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

class Config:
    DATA_ROOT = _resolve_data_root()

    # 上传文件配置
    UPLOAD_FOLDER = os.path.join(DATA_ROOT, 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'log', 'md', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'}
    
    # 向量数据库配置
    VECTOR_DB_PATH = os.path.join(DATA_ROOT, 'vector_db')
    RAG_CHUNK_SIZE = 800
    RAG_CHUNK_OVERLAP = 120
    RAG_TOP_K = 4
    RAG_BM25_MAX_DOCS = int(os.getenv('RAG_BM25_MAX_DOCS', '2000'))
    
    # 千问API配置
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-f9e461480cd64906a529f5d723749459')
    QWEN_BASE_URL = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen-plus')
    QWEN_EMBEDDING_MODEL = os.getenv('QWEN_EMBEDDING_MODEL', 'text-embedding-v3')
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_DIR = os.path.join(DATA_ROOT, 'logs')
    BACKEND_LOG_FILE = os.path.join(LOG_DIR, 'backend.log')
    
    # 其他配置
    SECRET_KEY = 'your-secret-key'

    # 自动清理配置
    AUTO_CLEANUP_ENABLED = os.getenv('AUTO_CLEANUP_ENABLED', 'true').lower() == 'true'
    AUTO_CLEANUP_EXPIRE_DAYS = int(os.getenv('AUTO_CLEANUP_EXPIRE_DAYS', '7'))
    AUTO_CLEANUP_INTERVAL_HOURS = int(os.getenv('AUTO_CLEANUP_INTERVAL_HOURS', '6'))
    
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # 确保向量数据库目录存在
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)