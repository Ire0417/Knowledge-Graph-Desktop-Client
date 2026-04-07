import importlib.util
import os
import sys

import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document


def _load_flask_app():
    app_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    spec = importlib.util.spec_from_file_location('backend_entry_app_multirecall', app_file)
    if spec is None or spec.loader is None:
        raise RuntimeError('Unable to load backend/app.py for testing')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def test_bm25_recall_hits_keyword_doc():
    from app.services.rag_service import _bm25_recall

    docs = [
        Document(page_content='北京大学位于北京。', metadata={'chunk_index': 0}),
        Document(page_content='清华大学在北京海淀区。', metadata={'chunk_index': 1}),
        Document(page_content='苹果是一种水果。', metadata={'chunk_index': 2}),
    ]

    hits = _bm25_recall('北京大学在哪里', docs, k=2)

    assert len(hits) >= 1
    assert '北京大学' in hits[0][0].page_content


def test_kg_structured_recall_from_graph_and_extract():
    from app.services.rag_service import _kg_structured_recall

    file_info = {
        'graph_result': {
            'nodes': [
                {'id': 'n1', 'name': '北京大学', 'type': 'ORG'},
                {'id': 'n2', 'name': '北京', 'type': 'LOCATION'},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relationship': '位于'},
            ],
        },
        'extract_result': {
            'entities': [{'text': '张三', 'type': 'PERSON'}],
            'relations': [{'subject': '张三', 'predicate': '任职于', 'object': '北京大学'}],
        },
    }

    kg = _kg_structured_recall('北京大学在哪里，张三和它有什么关系？', file_info, k=5)

    triples = kg.get('triples', [])
    assert triples
    assert any(t['subject'] == '北京大学' and t['object'] == '北京' for t in triples)
    assert any(t['subject'] == '张三' and t['object'] == '北京大学' for t in triples)


def test_faq_rule_recall_with_keywords():
    from app.services.rag_service import _faq_rule_recall

    file_info = {
        'faq_items': [
            {
                'question': '支持哪些文件格式？',
                'answer': '支持 pdf/docx/txt/md/xlsx 等格式。',
                'keywords': ['文件格式', '支持'],
            }
        ]
    }

    hits = _faq_rule_recall('系统支持什么文件格式？', file_info, k=2)

    assert len(hits) == 1
    assert '支持' in hits[0]['answer']


def test_rag_answer_fallback_when_no_material(monkeypatch):
    from app.services import rag_service

    app = _load_flask_app()
    with app.app_context():
        monkeypatch.setattr(rag_service, '_load_vector_store', lambda _file_id: {'persist_dir': 'mock', 'embeddings': None})
        monkeypatch.setattr(
            rag_service,
            '_load_simple_vector_store',
            lambda _persist, load_docs=True: {'docs': [], 'vectors': np.asarray([])},
        )
        monkeypatch.setattr(rag_service, '_similarity_search_indices', lambda **_kwargs: [])

        result = rag_service.rag_answer(
            question='完全无关的问题',
            file_id='f1',
            file_info={'rag_ready': True, 'parse_result': {}},
        )

    assert result['answer'] == '暂无相关知识，无法解答。'
    assert result['sources'] == []
