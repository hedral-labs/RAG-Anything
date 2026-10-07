#!/usr/bin/env python3
"""Local-only RAG-Anything runner for evidence-first research evaluation.

``index`` is deliberate, one-PDF index construction. ``query`` only reads an
already-complete index and emits validated physical-page evidence. The runner
uses RAG-Anything's MinerU content pipeline and LightRAG storage locally, with
local BGE embeddings. It does not start a persistent service or use a hosted
LLM/embedding endpoint. MinerU may start and stop its own loopback helper while
one indexing subprocess is active; no process survives the command.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import math
import numbers
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNNER_SCHEMA = "raganything-local-eval-runner-v1"
INDEX_SCHEMA = "raganything-local-eval-index-v1"
_INDEX_MANIFEST = "raganything-evaluation-index.json"
_SUMMARY = "summary.json"
_STORAGE = "storage"
_TEXT_CHUNKS = "kv_store_text_chunks.json"
_VECTOR_CHUNKS = "vdb_chunks.json"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_ABSTENTION = "INSUFFICIENT_EVIDENCE"


class RunnerInputError(ValueError):
    """A stable, secret-free request or local-index validation failure."""


@dataclass(frozen=True)
class IndexManifest:
    index_name: str
    document_id: str
    source_id: str
    source_sha256: str
    page_count: int
    embedding_model: str
    embedding_dimension: int
    index_backend: str
    parser: str
    manifest_sha256: str


@dataclass(frozen=True)
class GenerationConfig:
    model: Path
    context_count: int
    max_tokens: int


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def _nonblank(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunnerInputError(code)
    return value.strip()


def _safe_name(value: object, code: str) -> str:
    result = _nonblank(value, code)
    if _SAFE_NAME.fullmatch(result) is None:
        raise RunnerInputError(code)
    return result


def _sha256(value: object, code: str) -> str:
    result = _nonblank(value, code).lower()
    if _HEX_64.fullmatch(result) is None:
        raise RunnerInputError(code)
    return result


def _positive_int(value: object, code: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise RunnerInputError(code)
    return value


def _page_index(value: object, *, page_count: int, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < page_count:
        raise RunnerInputError(code)
    return value


def _root_path(value: object, code: str) -> Path:
    return Path(_nonblank(value, code)).expanduser()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RunnerInputError(code) from None


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _index_path(index_root: Path, index_name: str) -> Path:
    return index_root / index_name


def _physical_page_count(path: Path) -> int:
    try:
        import fitz

        document = fitz.open(path)
        try:
            count = document.page_count
        finally:
            document.close()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise RunnerInputError("input_document_invalid") from error
    return _positive_int(count, "input_document_invalid", maximum=20_000)


def _manifest(index_path: Path, *, expected_index_name: str | None = None) -> IndexManifest:
    manifest_path = index_path / _INDEX_MANIFEST
    raw = _read_json(manifest_path, "index_manifest_invalid")
    expected = {
        "schema_version",
        "index_name",
        "document_id",
        "source_id",
        "source_sha256",
        "page_count",
        "embedding_model",
        "embedding_dimension",
        "index_backend",
        "parser",
        "indexing_mode",
        "created_at",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise RunnerInputError("index_manifest_invalid")
    manifest = IndexManifest(
        index_name=_safe_name(raw.get("index_name"), "index_manifest_invalid"),
        document_id=_nonblank(raw.get("document_id"), "index_manifest_invalid"),
        source_id=_nonblank(raw.get("source_id"), "index_manifest_invalid"),
        source_sha256=_sha256(raw.get("source_sha256"), "index_manifest_invalid"),
        page_count=_positive_int(raw.get("page_count"), "index_manifest_invalid", maximum=20_000),
        embedding_model=_nonblank(raw.get("embedding_model"), "index_manifest_invalid"),
        embedding_dimension=_positive_int(raw.get("embedding_dimension"), "index_manifest_invalid", maximum=16_384),
        index_backend=_nonblank(raw.get("index_backend"), "index_manifest_invalid"),
        parser=_nonblank(raw.get("parser"), "index_manifest_invalid"),
        manifest_sha256=_file_sha256(manifest_path),
    )
    if (
        raw.get("schema_version") != INDEX_SCHEMA
        or raw.get("indexing_mode") != "raganything-mineru-lightrag-local-v1"
        or manifest.index_name != (expected_index_name or index_path.name)
        or manifest.index_backend != "lightrag-nano-vector"
        or manifest.parser != "mineru-3-local-pipeline"
    ):
        raise RunnerInputError("index_manifest_invalid")
    return manifest


def _text_chunks(index_path: Path, manifest: IndexManifest) -> dict[str, Mapping[str, object]]:
    raw = _read_json(index_path / _STORAGE / _TEXT_CHUNKS, "index_data_invalid")
    if not isinstance(raw, Mapping) or not raw:
        raise RunnerInputError("index_data_invalid")
    chunks: dict[str, Mapping[str, object]] = {}
    for chunk_id, row in raw.items():
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(row, Mapping):
            raise RunnerInputError("index_data_invalid")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RunnerInputError("index_data_invalid")
        _page_index(row.get("page_idx"), page_count=manifest.page_count, code="index_data_invalid")
        end = row.get("page_idx_end")
        if end is not None:
            _page_index(end, page_count=manifest.page_count, code="index_data_invalid")
        chunks[chunk_id] = row
    return chunks


def _tree_fingerprint(root: Path) -> dict[str, str]:
    """Return exact file content fingerprints to assert query-time immutability."""

    return {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_index(index_path: Path, *, expected_index_name: str | None = None) -> tuple[IndexManifest, dict[str, Mapping[str, object]]]:
    manifest = _manifest(index_path, expected_index_name=expected_index_name)
    required = (index_path / _SUMMARY, index_path / _STORAGE / _TEXT_CHUNKS, index_path / _STORAGE / _VECTOR_CHUNKS)
    if not index_path.is_dir() or not all(path.is_file() for path in required):
        raise RunnerInputError("index_not_found")
    summary = _read_json(index_path / _SUMMARY, "index_data_invalid")
    if (
        not isinstance(summary, Mapping)
        or set(summary) != {"schema_version", "document_id", "page_count", "text_chunk_count"}
        or summary.get("schema_version") != INDEX_SCHEMA
        or summary.get("document_id") != manifest.document_id
        or summary.get("page_count") != manifest.page_count
    ):
        raise RunnerInputError("index_data_invalid")
    chunks = _text_chunks(index_path, manifest)
    if summary.get("text_chunk_count") != len(chunks):
        raise RunnerInputError("index_data_invalid")
    return manifest, chunks


def _offline_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _embedding_function(*, model_path: Path, device: str):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RunnerInputError("embedding_runtime_unavailable") from error
    if not model_path.is_dir():
        raise RunnerInputError("embedding_model_not_local")
    _offline_environment()
    try:
        model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        dimension = int(model.get_sentence_embedding_dimension())
    except Exception as error:
        raise RunnerInputError("embedding_model_not_local") from error

    async def embed(texts: Sequence[str], **_kwargs: object):
        if not all(isinstance(text, str) for text in texts):
            raise RunnerInputError("embedding_input_invalid")
        return np.asarray(model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True))

    return embed, dimension


async def _empty_local_index_llm(_prompt: str, **_kwargs: object) -> str:
    """Disable graph extraction explicitly while retaining local text retrieval.

    The comparison lane uses MinerU parsing plus LightRAG's persisted chunk
    vector store. Graph/entity extraction is disabled rather than silently
    sending chunks to a hosted LLM; answer generation is a separate, local
    reader stage after retrieval.
    """

    return ""


async def _open_rag(*, storage: Path, parser_output: Path, embedding_model: Path, device: str, read_only: bool):
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        from raganything import RAGAnything, RAGAnythingConfig
    except ImportError as error:
        raise RunnerInputError("raganything_runtime_unavailable") from error
    embed, dimension = _embedding_function(model_path=embedding_model, device=device)
    config = RAGAnythingConfig(
        working_dir=str(storage),
        parser="mineru",
        parse_method="txt",
        parser_output_dir=str(parser_output),
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
        use_full_path=False,
    )
    lightrag = LightRAG(
        working_dir=str(storage),
        embedding_func=EmbeddingFunc(embedding_dim=dimension, max_token_size=512, func=embed),
        llm_model_func=_empty_local_index_llm,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        chunk_token_size=512,
        chunk_overlap_token_size=64,
        auto_manage_storages_states=False,
    )
    rag = RAGAnything(
        lightrag=lightrag,
        llm_model_func=_empty_local_index_llm,
        embedding_func=EmbeddingFunc(embedding_dim=dimension, max_token_size=512, func=embed),
        config=config,
    )
    initialized = await rag._ensure_lightrag_initialized()
    if not initialized or not initialized.get("success"):
        raise RunnerInputError("raganything_initialization_failed")
    return rag, dimension


def _atomic_replace(build_path: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            if not overwrite:
                raise RunnerInputError("index_already_exists")
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            target.replace(backup)
        build_path.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


async def _build_async(
    *,
    source: Path,
    index_root: Path,
    index_name: str,
    document_id: str,
    source_id: str,
    source_sha256: str,
    embedding_model: str,
    embedding_model_path: Path,
    device: str,
    overwrite: bool,
) -> dict[str, object]:
    if not source.is_file():
        raise RunnerInputError("input_document_missing")
    if _file_sha256(source) != source_sha256:
        raise RunnerInputError("input_document_checksum_mismatch")
    page_count = _physical_page_count(source)
    index_root.mkdir(parents=True, exist_ok=True)
    target = _index_path(index_root, index_name)
    if target.exists() and not overwrite:
        raise RunnerInputError("index_already_exists")
    build_path = Path(tempfile.mkdtemp(prefix=f".{index_name}.build-", dir=index_root))
    rag = None
    try:
        storage = build_path / _STORAGE
        parser_output = build_path / "mineru_output"
        rag, dimension = await _open_rag(
            storage=storage,
            parser_output=parser_output,
            embedding_model=embedding_model_path,
            device=device,
            read_only=False,
        )
        await rag.process_document_complete(
            file_path=str(source),
            output_dir=str(parser_output),
            parse_method="txt",
            display_stats=False,
            doc_id=document_id,
            backend="pipeline",
            device=device,
            formula=False,
            table=False,
        )
        await rag.finalize_storages()
        rag = None
        manifest = {
            "schema_version": INDEX_SCHEMA,
            "index_name": index_name,
            "document_id": document_id,
            "source_id": source_id,
            "source_sha256": source_sha256,
            "page_count": page_count,
            "embedding_model": embedding_model,
            "embedding_dimension": dimension,
            "index_backend": "lightrag-nano-vector",
            "parser": "mineru-3-local-pipeline",
            "indexing_mode": "raganything-mineru-lightrag-local-v1",
            "created_at": _utc_now(),
        }
        _write_json(build_path / _INDEX_MANIFEST, manifest)
        loaded_manifest = _manifest(build_path, expected_index_name=index_name)
        chunks = _text_chunks(build_path, loaded_manifest)
        _write_json(
            build_path / _SUMMARY,
            {
                "schema_version": INDEX_SCHEMA,
                "document_id": document_id,
                "page_count": page_count,
                "text_chunk_count": len(chunks),
            },
        )
        _validate_index(build_path, expected_index_name=index_name)
        _atomic_replace(build_path, target, overwrite=overwrite)
    except Exception:
        if build_path.exists():
            shutil.rmtree(build_path, ignore_errors=True)
        raise
    finally:
        if rag is not None:
            with contextlib.suppress(Exception):
                await rag.finalize_storages()
    return {
        "schema_version": RUNNER_SCHEMA,
        "outcome": "indexed",
        "index_name": index_name,
        "document_id": document_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "page_count": page_count,
        "index_schema": INDEX_SCHEMA,
    }


def _identity(manifest: IndexManifest) -> dict[str, object]:
    return {
        "index_name": manifest.index_name,
        "document_id": manifest.document_id,
        "source_id": manifest.source_id,
        "source_sha256": manifest.source_sha256,
        "page_count": manifest.page_count,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "index_backend": manifest.index_backend,
        "parser": manifest.parser,
        "index_manifest_sha256": manifest.manifest_sha256,
    }


def _request() -> dict[str, object]:
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError):
        raise RunnerInputError("invalid_json") from None
    if not isinstance(payload, dict):
        raise RunnerInputError("invalid_request")
    allowed = {
        "index_root",
        "index_name",
        "question",
        "top_k",
        "mode",
        "device",
        "embedding_model_path",
        "generation_model",
        "generation_top_k",
        "max_tokens",
    }
    required = {"index_root", "index_name", "question", "top_k", "mode", "device", "embedding_model_path"}
    if not required.issubset(payload) or set(payload).difference(allowed):
        raise RunnerInputError("invalid_request")
    return payload


def _generation_config(payload: Mapping[str, object], *, evidence_count: int) -> GenerationConfig:
    required = ("generation_model", "generation_top_k", "max_tokens")
    if any(key not in payload for key in required):
        raise RunnerInputError("generation_config_missing")
    model = _root_path(payload.get("generation_model"), "generation_model_missing")
    if not model.is_dir():
        raise RunnerInputError("generation_model_not_local")
    context_count = _positive_int(payload.get("generation_top_k"), "generation_top_k_invalid", maximum=5)
    if context_count > evidence_count:
        raise RunnerInputError("generation_top_k_invalid")
    return GenerationConfig(
        model=model,
        context_count=context_count,
        max_tokens=_positive_int(payload.get("max_tokens"), "max_tokens_invalid", maximum=4096),
    )


def _strip_reasoning(value: str) -> str:
    return re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL).strip()


def _generate_answer(*, question: str, chunks: Sequence[str], config: GenerationConfig, device: str) -> str:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RunnerInputError("generation_runtime_unavailable") from error
    _offline_environment()
    context = "\n\n".join(
        f"[Retrieved evidence {rank}]\n{chunk}" for rank, chunk in enumerate(chunks, start=1)
    )
    prompt = (
        "Answer using only the retrieved evidence below. If the evidence is insufficient, "
        f"reply with exactly {_ABSTENTION}. Do not reveal reasoning or emit <think> tags.\n\n"
        f"Question: {question}\n\n{context}"
    )
    model = None
    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(str(config.model), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(config.model), local_files_only=True, torch_dtype=dtype
        ).eval().to(device)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=config.max_tokens, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    except Exception as error:
        raise RunnerInputError("generation_failed") from error
    finally:
        if model is not None:
            del model
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    answer = _strip_reasoning(answer)
    if not answer:
        raise RunnerInputError("generation_empty")
    return answer


async def _query_async(payload: Mapping[str, object]) -> dict[str, object]:
    index_root = _root_path(payload.get("index_root"), "index_root_missing")
    index_name = _safe_name(payload.get("index_name"), "index_name_missing")
    question = _nonblank(payload.get("question"), "question_missing")
    top_k = _positive_int(payload.get("top_k"), "top_k_invalid", maximum=100)
    device = _nonblank(payload.get("device"), "device_missing")
    mode = payload.get("mode")
    if mode not in {"retriever_only", "full_rag"}:
        raise RunnerInputError("mode_unsupported")
    embedding_model_path = _root_path(payload.get("embedding_model_path"), "embedding_model_missing")
    index_path = _index_path(index_root, index_name)
    manifest, chunks_by_id = _validate_index(index_path)
    before_query = _tree_fingerprint(index_path)
    rag = None
    try:
        rag, dimension = await _open_rag(
            storage=index_path / _STORAGE,
            parser_output=index_path / "mineru_output",
            embedding_model=embedding_model_path,
            device=device,
            read_only=True,
        )
        if dimension != manifest.embedding_dimension:
            raise RunnerInputError("embedding_profile_mismatch")
        raw_hits = await rag.lightrag.chunks_vdb.query(question, top_k=min(manifest.page_count * 8, 500))
    finally:
        # Query processes intentionally do not finalize/persist any LightRAG state.
        # They exit after the read-only search rather than saving storage back to disk.
        del rag
    if _tree_fingerprint(index_path) != before_query:
        raise RunnerInputError("index_mutation_detected")
    if not isinstance(raw_hits, list):
        raise RunnerInputError("search_result_invalid")
    evidence: list[dict[str, object]] = []
    evidence_text: list[str] = []
    seen_pages: set[int] = set()
    for hit in raw_hits:
        if not isinstance(hit, Mapping):
            raise RunnerInputError("search_result_invalid")
        chunk_id = hit.get("id")
        score = hit.get("distance")
        if not isinstance(chunk_id, str) or chunk_id not in chunks_by_id:
            raise RunnerInputError("search_result_invalid")
        if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score):
            raise RunnerInputError("search_result_invalid")
        chunk = chunks_by_id[chunk_id]
        page = _page_index(chunk.get("page_idx"), page_count=manifest.page_count, code="search_result_invalid")
        if page in seen_pages:
            continue
        seen_pages.add(page)
        page_end = chunk.get("page_idx_end")
        if page_end is not None:
            page_end = _page_index(page_end, page_count=manifest.page_count, code="search_result_invalid")
        text = chunk.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RunnerInputError("search_result_invalid")
        evidence.append(
            {
                "rank": len(evidence) + 1,
                "document_id": manifest.document_id,
                "source_id": manifest.source_id,
                "pdf_page_index": page,
                "native_page": page + 1,
                "score": float(score),
                "chunk_id": chunk_id,
                "chunk_page_end": page_end,
            }
        )
        evidence_text.append(text.strip())
        if len(evidence) == top_k:
            break
    response: dict[str, object] = {
        "schema_version": RUNNER_SCHEMA,
        "outcome": "retrieval_only",
        "answer": None,
        "results": evidence,
        "generation": {"requested": False, "model": None, "evidence_ranks": []},
        "identity": _identity(manifest),
    }
    if mode == "retriever_only":
        return response
    if not evidence:
        response.update(outcome="abstained")
        response["generation"] = {"requested": True, "model": None, "evidence_ranks": []}
        return response
    config = _generation_config(payload, evidence_count=len(evidence))
    selected_text = evidence_text[:config.context_count]
    answer = _strip_reasoning(
        _generate_answer(question=question, chunks=selected_text, config=config, device=device)
    )
    abstained = answer.casefold() == _ABSTENTION.casefold()
    response.update(outcome="abstained" if abstained else "answered", answer=None if abstained else answer)
    response["generation"] = {
        "requested": True,
        "model": str(config.model),
        "evidence_ranks": list(range(1, config.context_count + 1)),
    }
    return response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("query", help="Read one JSON request from stdin and emit one JSON response")
    index = commands.add_parser("index", help="Build and atomically publish one local PDF index")
    index.add_argument("--input", required=True)
    index.add_argument("--index-root", required=True)
    index.add_argument("--index-name", required=True)
    index.add_argument("--document-id", required=True)
    index.add_argument("--source-id", required=True)
    index.add_argument("--source-sha256", required=True)
    index.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    index.add_argument("--embedding-model-path", required=True)
    index.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    index.add_argument("--overwrite", action="store_true")
    return parser


def _index(arguments: argparse.Namespace) -> dict[str, object]:
    return asyncio.run(
        _build_async(
            source=Path(arguments.input).expanduser(),
            index_root=Path(arguments.index_root).expanduser(),
            index_name=_safe_name(arguments.index_name, "index_name_invalid"),
            document_id=_nonblank(arguments.document_id, "document_id_missing"),
            source_id=_nonblank(arguments.source_id, "source_id_missing"),
            source_sha256=_sha256(arguments.source_sha256, "source_sha256_invalid"),
            embedding_model=_nonblank(arguments.embedding_model, "embedding_model_missing"),
            embedding_model_path=Path(arguments.embedding_model_path).expanduser(),
            device=_nonblank(arguments.device, "device_missing"),
            overwrite=bool(arguments.overwrite),
        )
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        # Dependencies log verbosely; protocol stdout must contain exactly one JSON object.
        with contextlib.redirect_stdout(sys.stderr):
            response = _index(arguments) if arguments.command == "index" else asyncio.run(_query_async(_request()))
    except RunnerInputError as error:
        _emit({"schema_version": RUNNER_SCHEMA, "outcome": "error", "error_code": str(error)})
        return 2
    except Exception:  # noqa: BLE001 - the CLI protocol must not leak model/runtime details.
        _emit({"schema_version": RUNNER_SCHEMA, "outcome": "error", "error_code": "runner_failed"})
        return 1
    _emit(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
