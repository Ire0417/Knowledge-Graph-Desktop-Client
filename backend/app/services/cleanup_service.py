import os
import shutil
import time
from typing import Any, Dict, List, Set


def _extract_file_id_from_upload_name(filename: str) -> str:
    if '_' not in filename:
        return ''
    return filename.split('_', 1)[0]


def cleanup_expired_storage(upload_folder: str, vector_db_path: str, expire_days: int = 3) -> Dict[str, Any]:
    """清理超过保留期的上传文件与向量索引目录。"""
    now = time.time()
    safe_expire_days = max(1, int(expire_days))
    cutoff_ts = now - safe_expire_days * 24 * 60 * 60

    removed_upload_files = 0
    removed_vector_dirs = 0
    removed_file_ids: Set[str] = set()

    if os.path.isdir(upload_folder):
        for name in os.listdir(upload_folder):
            path = os.path.join(upload_folder, name)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff_ts:
                    os.remove(path)
                    removed_upload_files += 1
                    file_id = _extract_file_id_from_upload_name(name)
                    if file_id:
                        removed_file_ids.add(file_id)
            except OSError:
                continue

    if os.path.isdir(vector_db_path):
        for name in os.listdir(vector_db_path):
            path = os.path.join(vector_db_path, name)
            if not os.path.isdir(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff_ts:
                    shutil.rmtree(path, ignore_errors=True)
                    removed_vector_dirs += 1
                    removed_file_ids.add(name)
            except OSError:
                continue

    return {
        'expire_days': safe_expire_days,
        'removed_upload_files': removed_upload_files,
        'removed_vector_dirs': removed_vector_dirs,
        'removed_file_ids': sorted(list(removed_file_ids)),
    }