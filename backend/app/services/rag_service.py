import json
import math
import os
import re
import shutil
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
from flask import current_app
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _get_config() -> Dict[str, Any]:
    cfg = current_app.config
    return {
        'vector_db_path': cfg.get('VECTOR_DB_PATH'),
        'chunk_size': cfg.get('RAG_CHUNK_SIZE', 800),
        'chunk_overlap': cfg.get('RAG_CHUNK_OVERLAP', 120),
        'top_k': cfg.get('RAG_TOP_K', 4),
        'bm25_top_k': cfg.get('RAG_BM25_TOP_K', 4),
        'bm25_max_docs': cfg.get('RAG_BM25_MAX_DOCS', 2000),
        'kg_top_k': cfg.get('RAG_KG_TOP_K', 8),
        'faq_top_k': cfg.get('RAG_FAQ_TOP_K', 3),
        'qwen_api_key': cfg.get('QWEN_API_KEY', ''),
        'qwen_base_url': cfg.get('QWEN_BASE_URL', ''),
        'qwen_model': cfg.get('QWEN_MODEL', 'qwen-plus'),
        'qwen_embedding_model': cfg.get('QWEN_EMBEDDING_MODEL', 'text-embedding-v3'),
    }


def _ensure_api_key(api_key: str) -> None:
    if not api_key:
        raise RuntimeError('QWEN_API_KEY is empty. Please set it in environment or config.')


def _ensure_vector_db_path(path: str) -> None:
    if not path:
        raise RuntimeError('VECTOR_DB_PATH is empty. Please set it in config.')
    os.makedirs(path, exist_ok=True)


def _get_embedding_model(config: Dict[str, Any]) -> DashScopeEmbeddings:
    _ensure_api_key(config['qwen_api_key'])
    return DashScopeEmbeddings(
        model=config['qwen_embedding_model'],
        dashscope_api_key=config['qwen_api_key'],
    )


def _get_llm(config: Dict[str, Any]) -> ChatOpenAI:
    _ensure_api_key(config['qwen_api_key'])
    return ChatOpenAI(
        model=config['qwen_model'],
        api_key=config['qwen_api_key'],
        base_url=config['qwen_base_url'],
        temperature=0.2,
    )


def _flatten_tables(tables: List[Any]) -> str:
    rows: List[str] = []
    for table in tables:
        table_data = table.get('data') if isinstance(table, dict) else table
        if not isinstance(table_data, list):
            continue
        for row in table_data:
            if isinstance(row, list):
                cells = [str(cell).strip() for cell in row if str(cell).strip()]
                if cells:
                    rows.append(' | '.join(cells))
            elif isinstance(row, str) and row.strip():
                rows.append(row.strip())
    return '\n'.join(rows)


def _build_documents(file_id: str, file_info: Dict[str, Any]) -> List[Document]:
    parse_result = file_info.get('parse_result', {}) or {}
    text_parts: List[str] = []

    base_text = str(parse_result.get('text', '') or '').strip()
    if base_text:
        text_parts.append(base_text)

    table_text = _flatten_tables(parse_result.get('tables', []) or [])
    if table_text:
        text_parts.append(table_text)

    image_texts = parse_result.get('image_texts', []) or []
    for image_text in image_texts:
        if isinstance(image_text, str) and image_text.strip():
            text_parts.append(image_text.strip())

    merged_text = '\n\n'.join(text_parts).strip()
    if not merged_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=current_app.config.get('RAG_CHUNK_SIZE', 800),
        chunk_overlap=current_app.config.get('RAG_CHUNK_OVERLAP', 120),
        separators=['\n\n', '\n', '。', '！', '？', '.', '!', '?', '；', ';', '，', ',', ' '],
    )

    chunks = splitter.split_text(merged_text)
    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    'file_id': file_id,
                    'file_name': file_info.get('name', ''),
                    'chunk_index': idx,
                },
            )
        )
    return docs


def _vector_dir(file_id: str) -> str:
    config = _get_config()
    _ensure_vector_db_path(config['vector_db_path'])
    return os.path.join(config['vector_db_path'], file_id)


def _simple_docs_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, 'docs.json')


def _simple_docs_jsonl_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, 'docs.jsonl')


def _simple_vectors_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, 'vectors.npy')


def _save_simple_vector_store(persist_dir: str, docs: List[Document], vectors: List[List[float]]) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    with open(_simple_docs_path(persist_dir), 'w', encoding='utf-8') as f:
        json.dump(
            [
                {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata or {},
                }
                for doc in docs
            ],
            f,
            ensure_ascii=False,
        )

    jsonl_path = _simple_docs_jsonl_path(persist_dir)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            payload = {
                'page_content': doc.page_content,
                'metadata': doc.metadata or {},
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write('\n')

    np.save(_simple_vectors_path(persist_dir), np.asarray(vectors, dtype=np.float32))


def _load_simple_vector_store(persist_dir: str, load_docs: bool = True) -> Dict[str, Any]:
    docs_path = _simple_docs_path(persist_dir)
    vectors_path = _simple_vectors_path(persist_dir)
    if not os.path.exists(docs_path) or not os.path.exists(vectors_path):
        raise RuntimeError('Simple vector store files are missing. Please parse the file again.')
    docs: List[Document] = []
    if load_docs:
        with open(docs_path, 'r', encoding='utf-8') as f:
            raw_docs = json.load(f)

        docs = [
            Document(
                page_content=item.get('page_content', ''),
                metadata=item.get('metadata', {}),
            )
            for item in raw_docs
        ]
    vectors = np.load(vectors_path, mmap_mode='r')
    return {'docs': docs, 'vectors': vectors}


def _load_docs_by_indices(persist_dir: str, indices: List[int]) -> List[Document]:
    if not indices:
        return []

    targets = {int(i) for i in indices}
    jsonl_path = _simple_docs_jsonl_path(persist_dir)
    if os.path.exists(jsonl_path):
        found: Dict[int, Document] = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx not in targets:
                    continue
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                found[idx] = Document(
                    page_content=item.get('page_content', ''),
                    metadata=item.get('metadata', {}),
                )
                if len(found) >= len(targets):
                    break
        return [found[i] for i in indices if i in found]

    store = _load_simple_vector_store(persist_dir, load_docs=True)
    docs = store.get('docs', []) or []
    return [docs[i] for i in indices if 0 <= i < len(docs)]


def _similarity_search_simple(
    embeddings: DashScopeEmbeddings,
    docs: List[Document],
    vectors: np.ndarray,
    question: str,
    k: int,
) -> List[Document]:

    if len(docs) == 0 or vectors.size == 0:
        return []

    query_vec = np.asarray(embeddings.embed_query(question), dtype=np.float32)
    doc_norms = np.linalg.norm(vectors, axis=1) + 1e-12
    query_norm = float(np.linalg.norm(query_vec) + 1e-12)
    scores = (vectors @ query_vec) / (doc_norms * query_norm)

    top_indices = np.argsort(-scores)[: max(1, int(k))]
    return [docs[int(i)] for i in top_indices]


def _similarity_search_indices(
    embeddings: DashScopeEmbeddings,
    vectors: np.ndarray,
    question: str,
    k: int,
) -> List[int]:

    if vectors.size == 0:
        return []

    query_vec = np.asarray(embeddings.embed_query(question), dtype=np.float32)
    doc_norms = np.linalg.norm(vectors, axis=1) + 1e-12
    query_norm = float(np.linalg.norm(query_vec) + 1e-12)
    scores = (vectors @ query_vec) / (doc_norms * query_norm)

    top_indices = np.argsort(-scores)[: max(1, int(k))]
    return [int(i) for i in top_indices]


def _normalize_for_dedupe(text: str) -> str:
    text = (text or '').strip().lower()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r"[，。；：！？,.!?;:\-()（）\[\]{}\"']+", '', text)
    return text


def _dedupe_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs: List[Document] = []
    for doc in docs:
        key = _normalize_for_dedupe(doc.page_content)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def _tokenize_for_search(text: str) -> List[str]:
    if not text:
        return []

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    tokens: List[str] = []
    for token in jieba.lcut(text):
        t = token.strip().lower()
        if not t:
            continue
        if len(t) == 1 and not t.isdigit() and not re.match(r'[a-zA-Z0-9]', t):
            continue
        tokens.append(t)

    for token in re.findall(r'[a-zA-Z0-9_]+', text):
        t = token.strip().lower()
        if t:
            tokens.append(t)

    return tokens


def _bm25_recall(question: str, docs: List[Document], k: int) -> List[Tuple[Document, float]]:
    if not docs:
        return []

    query_tokens = _tokenize_for_search(question)
    if not query_tokens:
        return []

    query_counter = Counter(query_tokens)
    doc_tokens = [_tokenize_for_search(doc.page_content) for doc in docs]
    doc_term_freqs = [Counter(tokens) for tokens in doc_tokens]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    if avg_doc_len <= 0:
        return []

    df_counter = Counter()
    for term_freq in doc_term_freqs:
        for token in term_freq:
            df_counter[token] += 1

    n_docs = len(docs)
    k1 = 1.5
    b = 0.75
    scored: List[Tuple[int, float]] = []

    for idx, term_freq in enumerate(doc_term_freqs):
        score = 0.0
        doc_len = max(1, doc_lengths[idx])
        for token, qtf in query_counter.items():
            tf = term_freq.get(token, 0)
            if tf <= 0:
                continue
            df = df_counter.get(token, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1)) / denom) * qtf
        if score > 0:
            scored.append((idx, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_hits = scored[: max(1, int(k))]
    return [(docs[idx], score) for idx, score in top_hits]


def _safe_snippet(text: str, limit: int = 220) -> str:
    clean = re.sub(r'\s+', ' ', (text or '').strip())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + '...'


def _extract_graph_triples(file_info: Dict[str, Any]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    seen = set()

    graph_result = file_info.get('graph_result', {}) or {}
    nodes = graph_result.get('nodes', []) or []
    node_name_map = {
        str(node.get('id', '')): str(node.get('name', '')).strip()
        for node in nodes
        if str(node.get('id', '')).strip() and str(node.get('name', '')).strip()
    }

    for edge in graph_result.get('edges', []) or []:
        subject = str(edge.get('source', '')).strip()
        obj = str(edge.get('target', '')).strip()
        predicate = str(edge.get('relationship', '') or edge.get('predicate', '') or edge.get('label', '')).strip()
        subject = node_name_map.get(subject, subject)
        obj = node_name_map.get(obj, obj)

        if not subject or not predicate or not obj:
            continue
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({'subject': subject, 'predicate': predicate, 'object': obj})

    extract_result = file_info.get('extract_result', {}) or {}
    for rel in extract_result.get('relations', []) or []:
        subject = str(rel.get('subject', '')).strip()
        predicate = str(rel.get('predicate', '')).strip()
        obj = str(rel.get('object', '')).strip()
        if not subject or not predicate or not obj:
            continue
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({'subject': subject, 'predicate': predicate, 'object': obj})

    return triples


def _extract_entity_attrs(file_info: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    attrs: Dict[str, Dict[str, str]] = {}

    graph_result = file_info.get('graph_result', {}) or {}
    for node in graph_result.get('nodes', []) or []:
        name = str(node.get('name', '')).strip()
        if not name:
            continue
        attrs[name] = {'type': str(node.get('type', 'ENTITY')).strip() or 'ENTITY'}

    extract_result = file_info.get('extract_result', {}) or {}
    for entity in extract_result.get('entities', []) or []:
        name = str(entity.get('text', '')).strip()
        if not name:
            continue
        attrs[name] = {
            'type': str(entity.get('type', attrs.get(name, {}).get('type', 'ENTITY'))).strip() or 'ENTITY'
        }

    return attrs


def _kg_structured_recall(question: str, file_info: Dict[str, Any], k: int) -> Dict[str, Any]:
    triples = _extract_graph_triples(file_info)
    entity_attrs = _extract_entity_attrs(file_info)
    if not triples and not entity_attrs:
        return {'triples': [], 'entity_attributes': [], 'matched_entities': []}

    question_tokens = set(_tokenize_for_search(question))
    matched_entities: List[Tuple[str, float]] = []

    for entity_name in entity_attrs:
        score = 0.0
        if entity_name and entity_name in question:
            score += 2.0
        entity_tokens = set(_tokenize_for_search(entity_name))
        score += 0.5 * len(entity_tokens & question_tokens)
        if score > 0:
            matched_entities.append((entity_name, score))

    matched_entities.sort(key=lambda item: item[1], reverse=True)
    primary_entities = {name for name, _ in matched_entities[: max(3, min(8, int(k)))]}

    triple_scores: List[Tuple[Dict[str, str], float]] = []
    for triple in triples:
        subject = triple['subject']
        obj = triple['object']
        predicate = triple['predicate']

        score = 0.0
        if subject in question:
            score += 1.8
        if obj in question:
            score += 1.8
        if predicate and predicate in question:
            score += 1.0
        if subject in primary_entities:
            score += 1.2
        if obj in primary_entities:
            score += 1.2

        triple_tokens = set(_tokenize_for_search(subject + ' ' + predicate + ' ' + obj))
        score += 0.25 * len(triple_tokens & question_tokens)

        if score > 0:
            triple_scores.append((triple, score))

    triple_scores.sort(key=lambda item: item[1], reverse=True)
    selected_triples = [item[0] for item in triple_scores[: max(1, int(k))]]

    selected_entity_names = set(primary_entities)
    for triple in selected_triples:
        selected_entity_names.add(triple['subject'])
        selected_entity_names.add(triple['object'])

    selected_attrs = []
    for name in selected_entity_names:
        if name in entity_attrs:
            selected_attrs.append({'entity': name, 'type': entity_attrs[name].get('type', 'ENTITY')})

    selected_attrs.sort(key=lambda item: item['entity'])
    matched_entity_names = [name for name, _ in matched_entities]

    return {
        'triples': selected_triples,
        'entity_attributes': selected_attrs,
        'matched_entities': matched_entity_names,
    }


def _collect_faq_items(file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = file_info.get('faq_items')
    if raw is None:
        raw = file_info.get('faq_rules')
    if raw is None:
        raw = file_info.get('faq_data')
    if raw is None:
        return []

    if isinstance(raw, dict):
        raw = raw.get('items', [])

    if not isinstance(raw, list):
        return []

    items: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            q = str(item.get('question', '')).strip()
            a = str(item.get('answer', '')).strip()
            keywords = item.get('keywords', []) or []
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            q = str(item[0]).strip()
            a = str(item[1]).strip()
            keywords = []
        else:
            continue

        if not q or not a:
            continue
        items.append({'question': q, 'answer': a, 'keywords': keywords})

    return items


def _faq_rule_recall(question: str, file_info: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    faq_items = _collect_faq_items(file_info)
    if not faq_items:
        return []

    q_tokens = set(_tokenize_for_search(question))
    scored: List[Tuple[Dict[str, Any], float]] = []

    for item in faq_items:
        faq_q = item['question']
        score = 0.0

        if question == faq_q:
            score += 6.0
        elif faq_q in question or question in faq_q:
            score += 3.5

        faq_tokens = set(_tokenize_for_search(faq_q))
        score += 0.5 * len(faq_tokens & q_tokens)

        for kw in item.get('keywords', []) or []:
            kw_text = str(kw).strip()
            if kw_text and kw_text in question:
                score += 1.0

        if score > 0:
            scored.append((item, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_hits = [item for item, _ in scored[: max(1, int(k))]]
    return top_hits


def _build_recall_summary(
    question: str,
    kg_result: Dict[str, Any],
    bm25_hits: List[Tuple[Document, float]],
    vector_docs: List[Document],
    faq_hits: List[Dict[str, Any]],
) -> str:
    kg_lines: List[str] = []
    triples = kg_result.get('triples', []) or []
    entity_attributes = kg_result.get('entity_attributes', []) or []

    if triples:
        kg_lines.append('三元组:')
        for triple in triples:
            kg_lines.append(
                f"- {triple.get('subject', '')} --{triple.get('predicate', '')}--> {triple.get('object', '')}"
            )
    if entity_attributes:
        kg_lines.append('实体属性:')
        for attr in entity_attributes:
            kg_lines.append(f"- {attr.get('entity', '')}: type={attr.get('type', 'ENTITY')}")
    if not kg_lines:
        kg_lines.append('无匹配')

    bm25_lines: List[str] = []
    if bm25_hits:
        for idx, (doc, score) in enumerate(bm25_hits, start=1):
            bm25_lines.append(f"{idx}. score={score:.4f} | {_safe_snippet(doc.page_content)}")
    else:
        bm25_lines.append('无匹配')

    vector_lines: List[str] = []
    if vector_docs:
        for idx, doc in enumerate(vector_docs, start=1):
            vector_lines.append(f"{idx}. {_safe_snippet(doc.page_content)}")
    else:
        vector_lines.append('无匹配')

    faq_lines: List[str] = []
    if faq_hits:
        for idx, item in enumerate(faq_hits, start=1):
            faq_lines.append(
                f"{idx}. 问: {item.get('question', '')} | 答: {_safe_snippet(item.get('answer', ''), limit=160)}"
            )
    else:
        faq_lines.append('无匹配')

    sections = [
        f"用户问题：{question.strip()}",
        '1) KG结构化召回',
        '\n'.join(kg_lines),
        '2) BM25关键词召回',
        '\n'.join(bm25_lines),
        '3) 向量语义召回',
        '\n'.join(vector_lines),
        '4) FAQ规则召回',
        '\n'.join(faq_lines),
    ]
    return '\n'.join(sections).strip()


def _build_sources(
    kg_result: Dict[str, Any],
    bm25_hits: List[Tuple[Document, float]],
    vector_docs: List[Document],
    faq_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []

    for triple in kg_result.get('triples', []) or []:
        sources.append(
            {
                'route': 'kg',
                'subject': triple.get('subject', ''),
                'predicate': triple.get('predicate', ''),
                'object': triple.get('object', ''),
            }
        )

    for doc, score in bm25_hits:
        payload = dict(doc.metadata or {})
        payload.update({'route': 'bm25', 'score': round(float(score), 6), 'snippet': _safe_snippet(doc.page_content)})
        sources.append(payload)

    for doc in vector_docs:
        payload = dict(doc.metadata or {})
        payload.update({'route': 'vector', 'snippet': _safe_snippet(doc.page_content)})
        sources.append(payload)

    for item in faq_hits:
        sources.append(
            {
                'route': 'faq',
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
            }
        )

    return sources


def build_file_vector_store(file_id: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """将解析内容切片并写入文件级向量库。"""
    docs = _build_documents(file_id, file_info)
    if not docs:
        raise RuntimeError('No valid content extracted from file for vectorization.')

    config = _get_config()
    persist_dir = _vector_dir(file_id)

    # 每次重新解析后重建该文件向量库，避免旧分片残留。
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

    try:
        embeddings = _get_embedding_model(config)
        vectors = embeddings.embed_documents([doc.page_content for doc in docs])
        _save_simple_vector_store(persist_dir, docs, vectors)
    except Exception as e:
        raise RuntimeError(f'Failed to build vector store: {str(e)}') from e

    file_info['rag_ready'] = True
    file_info['vector_store_path'] = persist_dir
    file_info['vector_chunk_count'] = len(docs)

    return {
        'vector_store_path': persist_dir,
        'chunk_count': len(docs),
    }


def delete_file_vector_store(file_id: str) -> None:
    """删除指定文件的向量库目录。"""
    persist_dir = _vector_dir(file_id)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def _load_vector_store(file_id: str) -> Dict[str, Any]:
    config = _get_config()
    persist_dir = _vector_dir(file_id)
    if not os.path.exists(persist_dir):
        raise RuntimeError('Vector store not found for this file. Please parse the file first.')

    try:
        embeddings = _get_embedding_model(config)
        return {'persist_dir': persist_dir, 'embeddings': embeddings}
    except Exception as e:
        raise RuntimeError(f'Failed to load vector store: {str(e)}') from e


def rag_answer(question: str, file_id: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """基于KG+多路召回RAG的千问问答。"""
    config = _get_config()

    if not file_info.get('rag_ready'):
        if file_info.get('parse_result'):
            try:
                build_file_vector_store(file_id, file_info)
            except Exception as e:
                raise RuntimeError(f'RAG index is not ready: {str(e)}') from e
        else:
            raise RuntimeError('File is not parsed yet. Please parse the file before asking questions.')

    vector_store = _load_vector_store(file_id)
    simple_store = _load_simple_vector_store(vector_store['persist_dir'], load_docs=False)
    all_vectors = simple_store.get('vectors', np.asarray([]))

    vector_docs: List[Document] = []
    bm25_hits: List[Tuple[Document, float]] = []

    if all_vectors.size > 0:
        try:
            bm25_max_docs = int(config.get('bm25_max_docs') or 0)
            candidate_k = max(int(config['top_k']), bm25_max_docs if bm25_max_docs > 0 else int(config['top_k']))
            candidate_indices = _similarity_search_indices(
                embeddings=vector_store['embeddings'],
                vectors=all_vectors,
                question=question,
                k=candidate_k,
            )
            vector_indices = candidate_indices[: max(1, int(config['top_k']))]
            vector_docs = _load_docs_by_indices(vector_store['persist_dir'], vector_indices)

            if bm25_max_docs > 0:
                bm25_indices = candidate_indices[: max(1, bm25_max_docs)]
                bm25_docs = _load_docs_by_indices(vector_store['persist_dir'], bm25_indices)
                bm25_hits = _bm25_recall(question, bm25_docs, config['bm25_top_k'])
            else:
                all_docs = _load_simple_vector_store(vector_store['persist_dir'], load_docs=True).get('docs', [])
                bm25_hits = _bm25_recall(question, all_docs, config['bm25_top_k'])
        except Exception as e:
            raise RuntimeError(f'Failed to retrieve relevant chunks: {str(e)}') from e
    else:
        all_docs = _load_simple_vector_store(vector_store['persist_dir'], load_docs=True).get('docs', [])
        bm25_hits = _bm25_recall(question, all_docs, config['bm25_top_k'])

    vector_docs = _dedupe_documents(vector_docs)
    kg_result = _kg_structured_recall(question, file_info, config['kg_top_k'])
    faq_hits = _faq_rule_recall(question, file_info, config['faq_top_k'])

    has_kg = bool(kg_result.get('triples') or kg_result.get('entity_attributes'))
    has_bm25 = bool(bm25_hits)
    has_vector = bool(vector_docs)
    has_faq = bool(faq_hits)

    if not (has_kg or has_bm25 or has_vector or has_faq):
        return {
            'answer': '暂无相关知识，无法解答。',
            'sources': [],
            'recall': {
                'kg': kg_result,
                'bm25_count': 0,
                'vector_count': 0,
                'faq_count': 0,
            },
        }

    recall_summary = _build_recall_summary(
        question=question,
        kg_result=kg_result,
        bm25_hits=bm25_hits,
        vector_docs=vector_docs,
        faq_hits=faq_hits,
    )

    llm = _get_llm(config)
    prompt = ChatPromptTemplate.from_template(
        """
角色
你是知识图谱多路召回RAG专属问答助手，严格依赖检索素材作答，禁止虚构内容。

素材来源（四路召回）
1. 向量语义召回：语义相似文本片段
2. BM25关键词召回：实体/术语精准匹配内容
3. KG结构化召回：三元组、实体属性、图谱关系路径
4. FAQ规则召回：高频固定标准答案

处理规则
1. 优先级：KG结构化 > BM25关键词 > 向量语义 > FAQ，冲突以图谱为准；
2. 自动去重、合并冗余信息，补全细节；
3. 无匹配素材直接说明：暂无相关知识，无法解答；
4. 回答简洁有条理，关系类清晰罗列实体关联，科普类分层说明。

输入格式
用户问题：{question}
四路召回数据：
{recall_summary}

输出要求
紧扣问题，只融合现有素材，逻辑清晰、不编造、不扩展额外知识。
""".strip()
    )
    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({'question': question, 'recall_summary': recall_summary})
    except Exception as e:
        raise RuntimeError(f'LLM invocation failed: {str(e)}') from e

    return {
        'answer': answer,
        'sources': _build_sources(
            kg_result=kg_result,
            bm25_hits=bm25_hits,
            vector_docs=vector_docs,
            faq_hits=faq_hits,
        ),
        'recall': {
            'kg': kg_result,
            'bm25_count': len(bm25_hits),
            'vector_count': len(vector_docs),
            'faq_count': len(faq_hits),
        },
    }


def check_rag_health(file_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """检查RAG可用性：配置、Embedding连通性、LLM连通性。"""
    config = _get_config()
    report: Dict[str, Any] = {
        'ok': True,
        'checks': {
            'api_key': {'ok': True, 'message': 'ok'},
            'vector_db_path': {'ok': True, 'message': 'ok'},
            'embedding': {'ok': True, 'message': 'ok'},
            'llm': {'ok': True, 'message': 'ok'},
        },
        'models': {
            'llm': config.get('qwen_model', ''),
            'embedding': config.get('qwen_embedding_model', ''),
        },
    }

    if file_info is not None:
        report['file'] = {
            'rag_ready': bool(file_info.get('rag_ready')),
            'status': file_info.get('status', ''),
            'rag_error': file_info.get('rag_error', ''),
        }

    try:
        _ensure_api_key(config['qwen_api_key'])
    except Exception as e:
        report['checks']['api_key'] = {'ok': False, 'message': str(e)}

    try:
        _ensure_vector_db_path(config['vector_db_path'])
    except Exception as e:
        report['checks']['vector_db_path'] = {'ok': False, 'message': str(e)}

    if report['checks']['api_key']['ok']:
        try:
            embeddings = _get_embedding_model(config)
            embeddings.embed_query('health check')
        except Exception as e:
            report['checks']['embedding'] = {'ok': False, 'message': str(e)}

        try:
            llm = _get_llm(config)
            llm.invoke('请仅回复: OK')
        except Exception as e:
            report['checks']['llm'] = {'ok': False, 'message': str(e)}
    else:
        report['checks']['embedding'] = {'ok': False, 'message': 'Skipped because API key is invalid'}
        report['checks']['llm'] = {'ok': False, 'message': 'Skipped because API key is invalid'}

    report['ok'] = all(item.get('ok') for item in report['checks'].values())
    return report
