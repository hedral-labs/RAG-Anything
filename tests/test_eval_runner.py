"""Contracts for the strict local RAG-Anything evaluation runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import eval_runner

_HASH = "a" * 64


def _index(tmp_path: Path) -> Path:
    index = tmp_path / "indexes" / "synthetic-index"
    storage = index / "storage"
    storage.mkdir(parents=True)
    manifest = {
        "schema_version": eval_runner.INDEX_SCHEMA,
        "index_name": index.name,
        "document_id": "synthetic-document",
        "source_id": "synthetic-source",
        "source_sha256": _HASH,
        "page_count": 4,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "index_backend": "lightrag-nano-vector",
        "parser": "mineru-3-local-pipeline",
        "indexing_mode": "raganything-mineru-lightrag-local-v1",
        "created_at": "2026-10-07T00:00:00Z",
    }
    (index / eval_runner._INDEX_MANIFEST).write_text(json.dumps(manifest))
    (index / eval_runner._SUMMARY).write_text(
        json.dumps(
            {
                "schema_version": eval_runner.INDEX_SCHEMA,
                "document_id": "synthetic-document",
                "page_count": 4,
                "text_chunk_count": 2,
            }
        )
    )
    (storage / eval_runner._TEXT_CHUNKS).write_text(
        json.dumps(
            {
                "chunk-a": {
                    "content": "The first tested paragraph.",
                    "page_idx": 1,
                    "page_idx_end": 1,
                },
                "chunk-b": {"content": "The second tested paragraph.", "page_idx": 3},
            }
        )
    )
    (storage / eval_runner._VECTOR_CHUNKS).write_text("{}")
    return index


class _FakeVdb:
    async def query(self, _question: str, *, top_k: int) -> list[dict[str, object]]:
        assert top_k == 32
        return [
            {"id": "chunk-a", "distance": np.float32(0.91)},
            {"id": "chunk-a", "distance": np.float32(0.90)},
            {"id": "chunk-b", "distance": np.float32(0.80)},
        ]


class _FakeRag:
    lightrag = type("LightRAG", (), {"chunks_vdb": _FakeVdb()})()

    async def finalize_storages(self) -> None:
        return None


async def _open_fake(**_kwargs: object) -> tuple[_FakeRag, int]:
    return _FakeRag(), 384


def _payload(index: Path, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "index_root": str(index.parent),
        "index_name": index.name,
        "question": "SYNTHETIC QUESTION",
        "top_k": 2,
        "mode": "retriever_only",
        "device": "cpu",
        "embedding_model_path": "/local/bge",
    }
    payload.update(changes)
    return payload


def test_query_returns_unique_canonical_physical_page_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = _index(tmp_path)
    monkeypatch.setattr(eval_runner, "_open_rag", _open_fake)

    response = asyncio.run(eval_runner._query_async(_payload(index)))

    assert response["outcome"] == "retrieval_only"
    assert response["results"] == [
        {
            "rank": 1,
            "document_id": "synthetic-document",
            "source_id": "synthetic-source",
            "pdf_page_index": 1,
            "native_page": 2,
            "score": 0.9100000262260437,
            "chunk_id": "chunk-a",
            "chunk_page_end": 1,
        },
        {
            "rank": 2,
            "document_id": "synthetic-document",
            "source_id": "synthetic-source",
            "pdf_page_index": 3,
            "native_page": 4,
            "score": 0.800000011920929,
            "chunk_id": "chunk-b",
            "chunk_page_end": None,
        },
    ]


def test_full_rag_uses_only_ranked_retrieval_text_and_strips_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _index(tmp_path)
    reader = tmp_path / "reader"
    reader.mkdir()
    received: dict[str, object] = {}

    def generate(**kwargs: object) -> str:
        received.update(kwargs)
        return "<think>hidden</think>grounded answer"

    monkeypatch.setattr(eval_runner, "_open_rag", _open_fake)
    monkeypatch.setattr(eval_runner, "_generate_answer", generate)
    response = asyncio.run(
        eval_runner._query_async(
            _payload(
                index,
                mode="full_rag",
                generation_model=str(reader),
                generation_top_k=1,
                max_tokens=128,
            )
        )
    )

    assert response["outcome"] == "answered"
    assert response["answer"] == "grounded answer"
    assert response["generation"] == {"requested": True, "model": str(reader), "evidence_ranks": [1]}
    assert received["chunks"] == ["The first tested paragraph."]


def test_query_rejects_any_index_content_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = _index(tmp_path)

    async def open_mutating(**kwargs: object) -> tuple[_FakeRag, int]:
        storage = kwargs["storage"]
        assert isinstance(storage, Path)
        (storage / eval_runner._VECTOR_CHUNKS).write_text('{"mutated":true}')
        return _FakeRag(), 384

    monkeypatch.setattr(eval_runner, "_open_rag", open_mutating)
    with pytest.raises(eval_runner.RunnerInputError, match="index_mutation_detected"):
        asyncio.run(eval_runner._query_async(_payload(index)))


def test_manifest_rejects_an_indexing_mode_that_is_not_the_local_contract(tmp_path: Path) -> None:
    index = _index(tmp_path)
    manifest = json.loads((index / eval_runner._INDEX_MANIFEST).read_text())
    manifest["indexing_mode"] = "unexpected"
    (index / eval_runner._INDEX_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(eval_runner.RunnerInputError, match="index_manifest_invalid"):
        eval_runner._validate_index(index)


def test_manifest_hash_is_the_exact_persisted_manifest(tmp_path: Path) -> None:
    index = _index(tmp_path)

    manifest, _ = eval_runner._validate_index(index)

    assert manifest.manifest_sha256 == hashlib.sha256(
        (index / eval_runner._INDEX_MANIFEST).read_bytes()
    ).hexdigest()


def test_generation_config_requires_a_real_local_reader_directory(tmp_path: Path) -> None:
    with pytest.raises(eval_runner.RunnerInputError, match="generation_model_not_local"):
        eval_runner._generation_config(
            {"generation_model": str(tmp_path / "missing"), "generation_top_k": 1, "max_tokens": 128},
            evidence_count=1,
        )
