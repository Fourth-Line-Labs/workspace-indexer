"""Building the store from settings."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from workspace_indexer.config import Settings
from workspace_indexer.models import EmbeddingSpace, SparseVec
from workspace_indexer.storage.qdrant_store import QdrantStore
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.store_factory import build_vector_store

SPACE = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)


async def test_embedded_mode_creates_its_data_directory(tmp_path: Path) -> None:
    """A missing parent would fail at the first upsert, deep into a run, rather
    than at startup."""
    target = tmp_path / "nested" / "deeper" / "qdrant"
    store = build_vector_store(Settings(qdrant_mode="embedded", qdrant_path=target), "labbox")
    assert target.parent.exists()
    await store.close()


async def test_embedded_mode_does_not_ask_for_payload_indexes(tmp_path: Path) -> None:
    """Local Qdrant ignores them and warns once per field, so asking anyway
    would emit eleven warnings on every single run."""
    store = build_vector_store(
        Settings(qdrant_mode="embedded", qdrant_path=tmp_path / "q"), "labbox"
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        await store.ensure_collection(EmbeddingSpace(model="fake:m", dimensions=4))
    assert not [w for w in captured if "Payload indexes" in str(w.message)]
    await store.close()


def test_server_mode_does_not_create_a_local_store(tmp_path: Path) -> None:
    """Pointing at a server must not quietly leave an unused local database
    behind, which would then hold a lock nobody expects."""
    local = tmp_path / "should_not_exist" / "qdrant"
    settings = Settings(qdrant_mode="server", qdrant_url="http://localhost:6333", qdrant_path=local)
    with warnings.catch_warnings():
        # No server is running here; the client warns it cannot check version.
        warnings.simplefilter("ignore")
        build_vector_store(settings, "labbox")
    assert not local.exists()


async def test_workspace_name_scopes_the_collection(tmp_path: Path) -> None:
    """Two workspaces on one Qdrant must not share a collection."""
    client = AsyncQdrantClient(path=str(tmp_path / "q"))
    a = QdrantStore(client, workspace="labbox", payload_indexes=False)
    b = QdrantStore(client, workspace="other", payload_indexes=False)
    assert a.collection_name(SPACE) == "labbox__voyageai_voyage-code-4_2048"
    assert b.collection_name(SPACE) != a.collection_name(SPACE)
    await client.close()


@pytest.mark.integration
async def test_server_mode_round_trip() -> None:
    """The plan's iteration-1 verification step, finally runnable.

    Needs Qdrant on :6333. Skips rather than fails when it is absent, so the
    suite still passes on a machine without one -- but it does exercise the
    server path, including the payload indexes that embedded mode silently
    ignores.
    """
    space = EmbeddingSpace(model="test:roundtrip", dimensions=4)
    try:
        client = AsyncQdrantClient(url="http://127.0.0.1:6333", timeout=5)
        await client.get_collections()
    except Exception:
        pytest.skip("no Qdrant server on :6333")

    store = QdrantStore(client, workspace="itest", payload_indexes=True)
    try:
        await store.drop_collection(space)
        await store.ensure_collection(space)

        # Payload indexes are the reason server mode matters for filtering;
        # embedded ignores them entirely.
        declared = await store.describe_vectors(space)
        assert declared == {"dense": ["dense"], "sparse": ["bm25"]}

        await store.upsert_points(
            space,
            ["11111111-1111-4111-8111-111111111111"],
            [[1.0, 0.0, 0.0, 0.0]],
            [SparseVec(indices=[7], values=[1.0])],
            [
                {
                    "rel_path": "a.py",
                    "root_label": "r",
                    "doc_type": "implementation",
                    "source_text": "def a(): pass",
                    "start_line": 1,
                    "end_line": 1,
                    "kind": "code",
                    "ancestors": [],
                }
            ],
        )
        assert await store.count(space) == 1

        hits = await store.search(space, QuerySpec(dense=[1.0, 0.0, 0.0, 0.0], fusion="dense_only"))
        assert [h.rel_path for h in hits] == ["a.py"]
        assert hits[0].doc_type.value == "implementation"
    finally:
        await store.drop_collection(space)
        await client.close()


def test_the_store_is_built_without_any_reranking_argument() -> None:
    """The factory used to take a `RerankConfig` solely to decide whether the
    store reranked. That implementation was retired in #73 and the refusal
    moved to config load, so a reranking argument here would be a second place
    to disagree about the same question."""
    from inspect import signature

    assert "rerank" not in signature(build_vector_store).parameters


def test_a_client_side_rerank_model_does_not_reach_the_store_at_all() -> None:
    """Reranking is now entirely a search-path concern: whichever client-side
    model is configured, the store is built the same way."""
    store = build_vector_store(Settings(vector_store="qdrant"), "w")
    assert store.describe().startswith("embedded qdrant")
