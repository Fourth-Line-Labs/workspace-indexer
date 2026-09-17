"""What the Mongo store actually sends to Atlas.

The assertions are about query documents, not results. That is deliberate and
the reasoning is in `fake_mongo_collection`: Atlas Search has no offline
equivalent, and every bug found in this store's first draft was a malformed
pipeline that any mocked *result* would have hidden. `test_mongo_integration`
runs the real thing when a connection string is configured.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from bson.binary import Binary, BinaryVectorDtype
from pymongo.errors import OperationFailure

from tests.fake_mongo_client import FakeMongoClient
from workspace_indexer.models import DocumentType, EmbeddingSpace, SearchFilters
from workspace_indexer.storage.mongo_index_spec import DENSE_FIELD, TEXT_INDEX, VECTOR_INDEX
from workspace_indexer.storage.mongo_store import (
    RRF_K,
    MongoStore,
    build_document,
    encode_vector,
)
from workspace_indexer.storage.query_spec import QuerySpec

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)


@pytest.fixture
def client() -> FakeMongoClient:
    return FakeMongoClient()


@pytest.fixture
def store(client: FakeMongoClient) -> MongoStore:
    return MongoStore(cast(Any, client), workspace="labbox", database="idx")


def collection(client: FakeMongoClient) -> Any:
    return client["idx"]["labbox__fake_model_4"]


def stages(pipeline: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(stage[name] for stage in pipeline if name in stage)


# ---- storage ---------------------------------------------------------------


def test_a_vector_is_stored_as_bindata_not_an_array_of_doubles() -> None:
    """The decision the whole Free-tier budget rests on: 4 bytes per dimension
    instead of BSON's 8-plus-key-overhead."""
    document = build_document("abc", [0.1, 0.2, 0.3, 0.4], {"rel_path": "a.py"})

    packed = document[DENSE_FIELD]
    assert isinstance(packed, Binary)
    assert packed.subtype == 9
    assert packed.as_vector().dtype is BinaryVectorDtype.FLOAT32


def test_bindata_is_materially_smaller_than_the_naive_encoding() -> None:
    """Measured, not asserted from the docs.

    On the live index this is 6.15 KB against 15.07 KB per document -- 66 MB
    against 162 MB for 11,049 chunks. This pins the ratio at the dimension
    count that matters so a change to the encoding cannot quietly give it back.
    """
    import bson

    vector = [0.01 * i for i in range(1024)]
    payload = {"rel_path": "src/a.py", "source_text": "x" * 1000}
    packed = len(bson.encode(build_document("id", vector, payload)))
    naive = len(bson.encode({**payload, "_id": "id", DENSE_FIELD: vector}))
    assert naive / packed > 1.8, f"binData saved less than expected: {naive} vs {packed}"


def test_int8_quantises_rather_than_handing_atlas_floats() -> None:
    packed = encode_vector([1.0, -1.0, 0.0], dtype="int8").as_vector()
    assert packed.dtype is BinaryVectorDtype.INT8
    assert list(packed.data) == [127, -127, 0]


def test_a_component_outside_the_int8_range_is_clamped_not_fatal() -> None:
    """`Binary.from_vector` raises outside [-128, 127]. A component
    fractionally over 1.0 must not fail a batch of 256 documents."""
    packed = encode_vector([1.4, -1.9], dtype="int8").as_vector()
    assert list(packed.data) == [127, -128]


def test_the_id_is_the_chunk_id_so_a_re_index_replaces_rather_than_duplicates() -> None:
    document = build_document("chunk-1", [0.0], {"rel_path": "a.py"})
    assert document["_id"] == "chunk-1"


# ---- schema ----------------------------------------------------------------


async def test_ensure_collection_creates_both_search_indexes(
    store: MongoStore, client: FakeMongoClient
) -> None:
    await store.ensure_collection(SPACE)
    created = {m.document["name"] for m in collection(client).created_search_indexes}
    assert created == {VECTOR_INDEX, TEXT_INDEX}


async def test_existing_search_indexes_are_not_recreated(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """Re-creating a search index rebuilds it, which on Atlas means an index
    that answers nothing until the rebuild finishes.

    A second store rather than clearing the first one's cache: a fresh process
    pointed at an already-indexed collection is the case that actually happens,
    and it is the one where an in-memory cache cannot help.
    """
    await store.ensure_collection(SPACE)
    fresh = MongoStore(cast(Any, client), workspace="labbox", database="idx")
    await fresh.ensure_collection(SPACE)
    assert len(collection(client).created_search_indexes) == 2


async def test_a_deployment_without_atlas_search_is_loud_but_not_fatal(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """Writing documents into a plain mongod is legitimate; discovering at
    query time that no search index exists is not."""
    import structlog

    collection(client).search_index_error = OperationFailure("no such command")
    with structlog.testing.capture_logs() as logs:
        await store.ensure_collection(SPACE)

    events = [entry["event"] for entry in logs]
    assert "store.search_indexes_unavailable" in events
    assert collection(client).created_search_indexes == []


# ---- query shapes ----------------------------------------------------------


async def test_dense_search_asks_for_the_vector_score_not_the_text_score(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """The two stages expose relevance under different metadata names. Asking
    for the wrong one is not an error -- it yields a missing field, so every
    hit comes back at 0.0 and the ranking silently collapses.
    """
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, fusion="dense_only", limit=5))

    pipeline = collection(client).pipelines[-1]
    assert stages(pipeline, "$vectorSearch")["index"] == VECTOR_INDEX
    assert stages(pipeline, "$addFields")["score"] == {"$meta": "vectorSearchScore"}


async def test_text_search_asks_for_the_search_score_and_limits(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """`$search` has no limit parameter of its own; without the `$limit` it
    streams the whole matching collection."""
    await store.search(SPACE, QuerySpec(text="auth", fusion="sparse_only", limit=7))

    pipeline = collection(client).pipelines[-1]
    assert stages(pipeline, "$search")["index"] == TEXT_INDEX
    assert stages(pipeline, "$limit") == 7
    assert stages(pipeline, "$addFields")["score"] == {"$meta": "searchScore"}


async def test_filters_are_pre_filters_in_both_branches(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """Never a `$match` after the search. Post-filtering a returned page
    silently shrinks the result set -- ask for 10 hits in one repo and get 3.
    Atlas makes this easy to get wrong, because the `$match` is valid.
    """
    filters = SearchFilters(repo_name="app", exclude_doc_types=[DocumentType.TEST])
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, fusion="dense_only"), filters)
    dense_pipeline = collection(client).pipelines[-1]
    assert stages(dense_pipeline, "$vectorSearch")["filter"] is not None
    assert not any("$match" in stage for stage in dense_pipeline)

    await store.search(SPACE, QuerySpec(text="auth", fusion="sparse_only"), filters)
    text_pipeline = collection(client).pipelines[-1]
    compound = stages(text_pipeline, "$search")["compound"]
    assert compound["filter"] == [{"equals": {"path": "repo_name", "value": "app"}}]
    assert compound["mustNot"] == [{"equals": {"path": "doc_type", "value": "test"}}]
    assert not any("$match" in stage for stage in text_pipeline)


async def test_rank_fusion_input_pipelines_carry_no_score_projection(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """`$rankFusion` accepts only ranked input pipelines and fuses on their
    order. A per-branch `$addFields` is not merely redundant -- it is what
    makes the stage reject the whole pipeline."""
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=5))

    fusion = stages(collection(client).pipelines[-1], "$rankFusion")
    pipelines = fusion["input"]["pipelines"]
    for branch in pipelines.values():
        assert not any("$addFields" in stage for stage in branch)
    assert "$vectorSearch" in pipelines["dense"][0]
    assert "$search" in pipelines["text"][0]


async def test_hybrid_falls_back_to_client_side_rrf_when_the_server_refuses(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """`$rankFusion` is a rolling deployment across the Atlas 8.0 fleet, so
    the only reliable question is the server's answer to it."""
    documents = [
        {"_id": "a", "rel_path": "a.py", "source_text": "a", "score": 1.0},
        {"_id": "b", "rel_path": "b.py", "source_text": "b", "score": 0.5},
    ]
    fake = collection(client)
    fake.documents = documents
    fake.unsupported_stage = "$rankFusion"

    hits = await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=5))

    assert [h.chunk_id for h in hits] == ["a", "b"]
    # Both branches ran once each, and both hits appeared in both, so the RRF
    # score is the sum of two reciprocal ranks rather than one.
    assert hits[0].score == pytest.approx(2 / (RRF_K + 1))


async def test_the_fallback_is_not_retried_once_it_is_known(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """One failed aggregation per process, not one per search."""
    fake = collection(client)
    fake.unsupported_stage = "$rankFusion"
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="a", limit=2))
    fake.pipelines.clear()

    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="a", limit=2))
    assert not any("$rankFusion" in stage for p in fake.pipelines for stage in p)


# ---- reading ---------------------------------------------------------------


async def test_chunks_for_path_matches_a_suffix_on_a_segment_boundary(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """`store.py` must not match `my_store.py`; a silently wrong file is worse
    than no answer."""
    fake = collection(client)
    fake.documents = [
        {"_id": "1", "rel_path": "app/src/store.py", "source_text": "right"},
        {"_id": "2", "rel_path": "app/src/my_store.py", "source_text": "wrong"},
    ]
    hits = await store.chunks_for_path(SPACE, "store.py")
    assert [h.rel_path for h in hits] == ["app/src/store.py"]


async def test_delete_by_path_targets_the_pair_not_the_path_alone(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """Two roots can hold the same relative path. Deleting on rel_path alone
    would take out a file in a repository nobody touched."""
    await store.delete_by_path(SPACE, "workspace", "app/src/store.py")
    assert collection(client).deleted[-1] == {
        "root_label": "workspace",
        "rel_path": "app/src/store.py",
    }


async def test_the_vector_is_not_returned_as_part_of_the_payload(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """`source_text` is what a hit carries; a 1024-float array in every result
    would be the largest thing in the response and useful to nobody."""
    collection(client).documents = [
        {"_id": "1", "rel_path": "a.py", "source_text": "x", DENSE_FIELD: encode_vector([1.0])}
    ]
    hits = await store.chunks_for_path(SPACE, "a.py")
    assert hits[0].source_text == "x"
    assert DENSE_FIELD not in hits[0].model_dump()


async def test_describe_names_the_backend_for_a_human(store: MongoStore) -> None:
    """Preflight used to reconstruct this from Qdrant settings, which reported
    an empty Mongo index as an empty Qdrant one."""
    assert store.describe() == "mongodb idx"


def test_the_document_we_build_is_within_the_bson_limit() -> None:
    """16 MB per document. A chunk is nowhere near it, but the failure mode if
    one ever were is a rejected write in the middle of a paid run."""
    import bson

    document = build_document("id", [0.1] * 2048, {"source_text": "x" * 100_000})
    assert len(bson.encode(document)) < 16 * 1024 * 1024


# ---- nothing reranks inside the query (#73) ---------------------------------


async def test_no_pipeline_the_store_builds_contains_a_rerank_stage(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """The retirement, asserted where it can actually break. `$rerank` is the
    one stage no reachable environment accepts -- absent from the local binary
    at every image tag, and Flex cannot reach the 8.3 it needs -- so a pipeline
    that grows one again fails every search against every cluster we have.
    """
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=8))
    collection(client).unsupported_stage = "$rankFusion"
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=8))

    built = collection(client).pipelines
    assert built, "no pipeline was captured, so this asserts nothing"
    assert not any("$rerank" in stage for pipeline in built for stage in pipeline)


async def test_the_candidate_set_is_not_widened(store: MongoStore, client: FakeMongoClient) -> None:
    """With nothing reordering results inside the query, retrieving fifty
    documents to return eight would be paid for and thrown away."""
    await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=8))

    limits = [s["$limit"] for s in collection(client).pipelines[-1] if "$limit" in s]
    assert limits[-1] == 8
    assert 50 not in limits


async def test_describe_names_the_database_and_nothing_else(client: FakeMongoClient) -> None:
    """Preflight prints this. It carried a `rerank=` suffix while the store
    could rerank; there is no second state to distinguish now."""
    assert MongoStore(cast(Any, client), workspace="w", database="idx").describe() == "mongodb idx"


async def test_a_rejected_fusion_stage_falls_back_rather_than_raising(
    store: MongoStore, client: FakeMongoClient
) -> None:
    """The guard that raised here existed only for database reranking: the
    client-side fallback fuses in Python, leaving no aggregation to append
    `$rerank` to. With no reranker in the store the fallback is a plain
    round-trip trade, so a deployment without `$rankFusion` must search, not
    fail."""
    collection(client).unsupported_stage = "$rankFusion"

    hits = await store.search(SPACE, QuerySpec(dense=[1.0] * 4, text="auth", limit=8))

    assert isinstance(hits, list)
