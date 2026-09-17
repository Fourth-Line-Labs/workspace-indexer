"""A VectorStore over MongoDB Atlas.

The second implementation of the storage seam, and the point of having had a
protocol at all: nothing above `storage/` changes to run on it. The payload is
built by the same `to_payload` and read back by the same `to_search_hit` as
Qdrant, so a chunk means the same thing on either backend and the eval numbers
from one are comparable with the other.

Three things genuinely differ, and each is a deliberate decision rather than an
accident of the driver:

**The sparse half is Lucene, not fastembed.** Qdrant stores a BM25 sparse
vector we compute locally; Atlas has its own inverted index and no way to
accept ours. So `upsert` takes the sparse vector and ignores it, and the
keyword branch of hybrid search is a `$search` stage over the same text. This
is not a downgrade -- Lucene computes IDF across the collection the same way
Qdrant's `Modifier.IDF` does -- but it is a different implementation of the
same idea, and a recall difference between the backends should be read with
that in mind rather than blamed on the vectors.

**Vectors are stored as BSON binData, not arrays of doubles.** Measured on this
workspace's own index -- 11,049 chunks at 1024 dimensions -- the naive array
encoding costs 15.07 KB per document against 6.15 KB for binData float32:
162 MB versus 66 MB. Shared tiers cap total storage -- 512 MB on Free, 5 GB on
Flex, both counting indexes -- so on Free that difference is most of the budget
and on Flex it is the difference between comfortable and careful.

**Fusion is server-side where the cluster allows it.** `$rankFusion` implements
RRF natively, but it is a rolling deployment across the Atlas 8.0 fleet. Rather
than guess from a version number, the first hybrid search tries it and falls
back to fusing client-side if the server rejects the stage. The fallback is the
same RRF formula, so results agree; it just costs two round trips instead of
one.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from bson.binary import Binary, BinaryVectorDtype
from pymongo import AsyncMongoClient, ReplaceOne
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from workspace_indexer.models import Chunk, EmbeddingSpace, SearchFilters, SearchHit, SparseVec
from workspace_indexer.obs.logging import get_logger, log_once
from workspace_indexer.storage.mongo_filter import match_stage, search_clauses, vector_filter
from workspace_indexer.storage.mongo_index_spec import (
    DENSE_FIELD,
    TEXT_FIELDS,
    TEXT_INDEX,
    VECTOR_INDEX,
    text_index,
    vector_index,
)
from workspace_indexer.storage.no_server_rerank import NoServerRerank
from workspace_indexer.storage.payload import to_payload, to_search_hit
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.server_reranker import ServerReranker

log = get_logger("workspace_indexer.storage.mongo")

# The RRF constant. 60 is the value from the original paper and the one both
# Qdrant and Atlas use, so the client-side fallback below and `$rankFusion`
# rank identically rather than merely similarly.
RRF_K = 60

_DTYPES = {
    "float32": BinaryVectorDtype.FLOAT32,
    "int8": BinaryVectorDtype.INT8,
}


class MongoStore:
    def __init__(
        self,
        client: AsyncMongoClient[dict[str, Any]],
        *,
        workspace: str,
        database: str,
        dtype: str = "float32",
        prefer_rank_fusion: bool = True,
    ) -> None:
        self._client = client
        self._workspace = workspace
        self._db = client[database]
        self._dtype = dtype
        self._ensured: set[str] = set()
        # An object rather than a flag, so nothing in the search path asks
        # whether to rerank -- it appends whatever tail it is given, and this
        # implementation gives it the plain scoring projection. The only other
        # implementation reranked inside the query and was retired in #73;
        # the seam is what #94 plugs back into, and it runs on every search
        # either way.
        self._rerank: ServerReranker = NoServerRerank()
        # None means "not yet discovered". Set on the first hybrid search, and
        # only ever set to False by the server actually rejecting the stage --
        # never inferred from a version string, because the rollout that gates
        # it does not line up with version numbers.
        # False here means "do not even try", which is different from the
        # discovered False below. Exists so the fallback can be exercised
        # against a cluster that *does* have the stage -- otherwise the only
        # way to test that path on real data is to find a server without it.
        self._rank_fusion: bool | None = None if prefer_rank_fusion else False
        # Whether this deployment has mongot at all. A plain community mongod
        # accepts documents happily and then fails every search, which is the
        # worst possible time to find out.
        self._search_indexes: bool | None = None

    def describe(self) -> str:
        return f"mongodb {self._db.name}"

    def collection_name(self, space: EmbeddingSpace) -> str:
        """The same name Qdrant would use, so a workspace indexed into both
        backends is recognisably the same collection in each."""
        return f"{self._workspace}__{space.slug()}"

    # ---- schema --------------------------------------------------------

    async def ensure_collection(self, space: EmbeddingSpace) -> None:
        name = self.collection_name(space)
        if name in self._ensured:
            return
        if name not in await self._db.list_collection_names():
            await self._db.create_collection(name)
            log.info("store.collection_created", collection=name, dimensions=space.dimensions)

        collection = self._db[name]
        # Ordinary b-tree indexes, unrelated to search. These are what make
        # delete_by_path and chunks_for_path a lookup rather than a scan of
        # every chunk in the workspace.
        await collection.create_index([("root_label", 1), ("rel_path", 1)])
        await collection.create_index([("space_slug", 1)])

        await self._ensure_search_indexes(name, space)
        self._ensured.add(name)

    async def _ensure_search_indexes(self, name: str, space: EmbeddingSpace) -> None:
        """Create the two mongot indexes, or say clearly why we cannot.

        Deliberately not fatal. Indexing into a deployment without Atlas Search
        is a legitimate thing to do -- the documents are correct and the
        indexes can be added later -- but it must be loud, because the failure
        it produces otherwise is a search that errors at query time, long after
        the run that would have explained it.
        """
        collection = self._db[name]
        try:
            existing = {index["name"] async for index in await collection.list_search_indexes()}
        except OperationFailure as exc:
            self._search_indexes = False
            log_once(
                log,
                f"mongo:no-search:{name}",
                "store.search_indexes_unavailable",
                collection=name,
                error=str(exc),
                detail=(
                    "this deployment has no Atlas Search; documents will be written "
                    "correctly but $vectorSearch and $search will fail at query time"
                ),
            )
            return

        self._search_indexes = True
        wanted = [
            SearchIndexModel(
                definition=vector_index(space.dimensions, dtype=self._dtype),
                name=VECTOR_INDEX,
                type="vectorSearch",
            ),
            SearchIndexModel(definition=text_index(), name=TEXT_INDEX, type="search"),
        ]
        missing = [model for model in wanted if model.document["name"] not in existing]
        if not missing:
            return
        await collection.create_search_indexes(missing)
        log.info(
            "store.search_indexes_created",
            collection=name,
            created=[m.document["name"] for m in missing],
            detail="built asynchronously; queries return nothing until they are ready",
        )

    # ---- writing -------------------------------------------------------

    async def upsert(
        self,
        space: EmbeddingSpace,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
    ) -> None:
        await self.upsert_points(
            space,
            [chunk.chunk_id for chunk in chunks],
            dense,
            sparse,
            [to_payload(chunk, space) for chunk in chunks],
        )

    async def upsert_points(
        self,
        space: EmbeddingSpace,
        ids: Sequence[str],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
        payloads: Sequence[dict[str, object]],
    ) -> None:
        if not ids:
            return
        if sparse:
            log_once(
                log,
                "mongo:sparse-ignored",
                "store.sparse_vectors_ignored",
                detail=(
                    "Atlas builds its own inverted index; the keyword branch of "
                    "hybrid search is a $search stage over source_text, not the "
                    "fastembed sparse vector"
                ),
            )
        await self.ensure_collection(space)
        collection = self._db[self.collection_name(space)]

        documents = [
            build_document(chunk_id, vector, payload, dtype=self._dtype)
            for chunk_id, vector, payload in zip(ids, dense, payloads, strict=True)
        ]
        operations = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in documents]
        # Unordered so one rejected document does not abandon the rest of the
        # batch; a partial write we know about beats a batch that stopped at
        # document three and reported success for nothing.
        result = await collection.bulk_write(operations, ordered=False)
        log.info(
            "store.upsert",
            collection=collection.name,
            count=len(operations),
            inserted=result.upserted_count,
            modified=result.modified_count,
        )

    async def delete_by_ids(self, space: EmbeddingSpace, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        collection = self._db[self.collection_name(space)]
        result = await collection.delete_many({"_id": {"$in": list(chunk_ids)}})
        log.info("store.delete", collection=collection.name, count=result.deleted_count)

    async def delete_by_path(self, space: EmbeddingSpace, root_label: str, rel_path: str) -> None:
        """What makes a deleted or renamed file correct without a manifest
        lookup: everything at that path goes, whatever chunks it produced."""
        collection = self._db[self.collection_name(space)]
        result = await collection.delete_many({"root_label": root_label, "rel_path": rel_path})
        log.info(
            "store.delete_by_path",
            collection=collection.name,
            rel_path=rel_path,
            count=result.deleted_count,
        )

    # ---- searching -----------------------------------------------------

    async def search(
        self, space: EmbeddingSpace, query: QuerySpec, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        name = self.collection_name(space)
        collection = self._db[name]

        # Retrieve deep, return shallow -- but only when something here is
        # going to reorder. With no server-side reranker this is `query.limit`
        # and the pipeline is exactly what it always was.
        depth = self._rerank.depth(query.limit)

        has_text = bool(query.text)
        if query.fusion == "dense_only" or not has_text:
            if query.dense is None:
                return []
            hits = await self._run(collection, self._dense_pipeline(query, filters, depth))
        elif query.fusion == "sparse_only" or query.dense is None:
            hits = await self._run(collection, self._text_pipeline(query, filters, depth))
        else:
            hits = await self._hybrid(collection, query, filters, depth)

        log.info(
            "search.store",
            collection=name,
            fusion=query.fusion,
            returned=len(hits),
            candidates=depth,
            reranker=self._rerank.name,
            filtered=filters is not None and not filters.is_empty(),
        )
        return hits

    async def _hybrid(
        self,
        collection: Any,
        query: QuerySpec,
        filters: SearchFilters | None,
        depth: int,
    ) -> list[SearchHit]:
        """Server-side RRF when the cluster has it, client-side when it does not.

        Tried rather than version-checked. `$rankFusion` is being rolled out
        across the Atlas 8.0 fleet on a schedule that does not correspond to
        any version string we can read, so asking the server is the only
        reliable question -- and asking it costs one failed aggregation, once
        per process.
        """
        if self._rank_fusion is not False:
            try:
                hits = await self._run(collection, self._rank_fusion_stages(query, filters, depth))
            except OperationFailure as exc:
                # Nothing else in this pipeline can be rejected now that the
                # store appends no reranking stage, so a failure here is the
                # fusion stage. When #94 restores server-side reranking this
                # has to narrow again: a refused `$rerank` was misdiagnosed as
                # a missing `$rankFusion` the first time it happened.
                self._rank_fusion = False
                log_once(
                    log,
                    "mongo:no-rank-fusion",
                    "store.rank_fusion_unavailable",
                    error=str(exc),
                    detail="fusing ranks client-side instead; same RRF, one extra round trip",
                )
            else:
                self._rank_fusion = True
                return hits
        return await self._client_side_fusion(collection, query, filters)

    def _rank_fusion_stages(
        self, query: QuerySpec, filters: SearchFilters | None, depth: int
    ) -> list[dict[str, Any]]:
        """Input pipelines carry no score projection of their own.

        `$rankFusion` accepts only *ranked* input pipelines, and it fuses on
        the order they produce rather than on any score inside them -- so the
        per-branch scores are not merely unnecessary here, adding them is what
        makes the stage reject the pipeline. The fused score comes out as
        `$meta: "score"` on the outside.
        """
        return [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "dense": self._dense_stages(query, filters, query.prefetch_limit),
                            "text": self._text_stages(query, filters, query.prefetch_limit),
                        }
                    }
                }
            },
            {"$limit": depth},
            *self._rerank.stages(query.text, query.limit, "score"),
        ]

    async def _client_side_fusion(
        self,
        collection: Any,
        query: QuerySpec,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        """The same RRF formula `$rankFusion` applies, computed here.

        Fusing on rank rather than score is not a simplification: cosine
        similarity and BM25 are not on a comparable scale, and averaging them
        produces an ordering that means nothing.
        """
        # Plain tails on both branches: this path fuses the two orderings
        # itself, so a per-branch rerank would reorder inputs the fusion is
        # about to reorder again.
        plain = NoServerRerank()
        dense = await self._run(
            collection, self._dense_pipeline(query, filters, query.prefetch_limit, plain)
        )
        text = await self._run(
            collection, self._text_pipeline(query, filters, query.prefetch_limit, plain)
        )

        scores: dict[str, float] = {}
        found: dict[str, SearchHit] = {}
        for branch in (dense, text):
            for rank, hit in enumerate(branch, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                found.setdefault(hit.chunk_id, hit)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: query.limit]
        results: list[SearchHit] = []
        for chunk_id, score in ordered:
            hit = found[chunk_id]
            results.append(hit.model_copy(update={"score": score}))
        return results

    def _dense_stages(
        self, query: QuerySpec, filters: SearchFilters | None, limit: int
    ) -> list[dict[str, Any]]:
        stage: dict[str, Any] = {
            "index": VECTOR_INDEX,
            "path": DENSE_FIELD,
            "queryVector": list(query.dense or []),
            # Candidates the ANN search considers before returning `limit`.
            # Atlas recommends well above the limit; too few and recall drops
            # in a way that looks like a bad embedding.
            "numCandidates": max(limit * 10, 100),
            "limit": limit,
        }
        condition = vector_filter(filters)
        if condition:
            stage["filter"] = condition
        return [{"$vectorSearch": stage}]

    def _dense_pipeline(
        self,
        query: QuerySpec,
        filters: SearchFilters | None,
        depth: int,
        tail: ServerReranker | None = None,
    ) -> list[dict[str, Any]]:
        # `vectorSearchScore`, not `searchScore`. The two stages expose their
        # relevance under different metadata names, and asking for the wrong
        # one is not an error -- it yields a missing field, so every hit comes
        # back scored 0.0 and the ranking silently collapses.
        return [
            *self._dense_stages(query, filters, depth),
            *(tail or self._rerank).stages(query.text, query.limit, "vectorSearchScore"),
        ]

    def _text_pipeline(
        self,
        query: QuerySpec,
        filters: SearchFilters | None,
        depth: int,
        tail: ServerReranker | None = None,
    ) -> list[dict[str, Any]]:
        return [
            *self._text_stages(query, filters, depth),
            *(tail or self._rerank).stages(query.text, query.limit, "searchScore"),
        ]

    def _text_stages(
        self, query: QuerySpec, filters: SearchFilters | None, limit: int
    ) -> list[dict[str, Any]]:
        include, exclude = search_clauses(filters)
        compound: dict[str, Any] = {
            "must": [{"text": {"query": query.text, "path": list(TEXT_FIELDS)}}]
        }
        if include:
            compound["filter"] = include
        if exclude:
            compound["mustNot"] = exclude
        # `$search` has no limit parameter of its own, unlike `$vectorSearch`.
        # Without the `$limit` it streams the whole matching collection.
        return [{"$search": {"index": TEXT_INDEX, "compound": compound}}, {"$limit": limit}]

    async def _run(self, collection: Any, stages: list[dict[str, Any]]) -> list[SearchHit]:
        # No translation layer: an `OperationFailure` propagates as itself. The
        # one message worth rewriting was Atlas's refusal of `$rerank`, and no
        # pipeline built here contains that stage any more (#73). #94 restores
        # both together or neither -- a raw "$rerank is not allowed or the
        # syntax is incorrect" sends the reader hunting their own pipeline for
        # a fault that is not in it.
        cursor = await collection.aggregate(stages)
        return [_to_hit(document) async for document in cursor]

    # ---- reading -------------------------------------------------------

    async def describe_vectors(self, space: EmbeddingSpace) -> dict[str, list[str]]:
        """Which search indexes exist and are queryable.

        Reported in the same shape Qdrant uses so `status` reads identically on
        either backend. The sparse list is the text index, because that is what
        plays the sparse half's role here -- an empty one means hybrid search
        is silently running dense-only.
        """
        collection = self._db[self.collection_name(space)]
        try:
            indexes = [index async for index in await collection.list_search_indexes()]
        except OperationFailure:
            return {"dense": [], "sparse": []}
        ready = {i["name"] for i in indexes if i.get("queryable")}
        return {
            "dense": [VECTOR_INDEX] if VECTOR_INDEX in ready else [],
            "sparse": [TEXT_INDEX] if TEXT_INDEX in ready else [],
        }

    async def count(self, space: EmbeddingSpace, filters: SearchFilters | None = None) -> int:
        name = self.collection_name(space)
        if name not in await self._db.list_collection_names():
            return 0
        return await self._db[name].count_documents(match_stage(filters))

    async def facet(self, space: EmbeddingSpace, key: str, limit: int = 32) -> dict[str, int]:
        name = self.collection_name(space)
        if name not in await self._db.list_collection_names():
            return {}
        cursor = await self._db[name].aggregate(
            [
                {"$group": {"_id": f"${key}", "n": {"$sum": 1}}},
                {"$sort": {"n": -1, "_id": 1}},
                {"$limit": limit},
            ]
        )
        # A null bucket is what "this field is unset" looks like after $group,
        # and reporting it as the string "None" would invent a value.
        return {str(d["_id"]): int(d["n"]) async for d in cursor if d["_id"] is not None}

    async def sample_paths(
        self, space: EmbeddingSpace, filters: SearchFilters | None = None, limit: int = 3
    ) -> list[str]:
        """A few distinct paths matching a filter.

        Distinct in the aggregation rather than after it: chunks cluster by
        file, so the first three documents are very often three chunks of one
        document, which makes a useless example set.
        """
        name = self.collection_name(space)
        if name not in await self._db.list_collection_names():
            return []
        stages: list[dict[str, Any]] = []
        match = match_stage(filters)
        if match:
            stages.append({"$match": match})
        stages += [
            {"$group": {"_id": "$rel_path"}},
            {"$sort": {"_id": 1}},
            {"$limit": limit},
        ]
        cursor = await self._db[name].aggregate(stages)
        return [str(d["_id"]) async for d in cursor]

    async def chunks_for_path(
        self, space: EmbeddingSpace, rel_path: str, limit: int = 50
    ) -> list[SearchHit]:
        """Every indexed chunk of one file, exact match first.

        The caller usually got `rel_path` from a search result, where it is the
        stored value verbatim. A model that typed it itself tends to give a
        suffix, so a miss falls back to an anchored regex on a segment
        boundary -- `store.py` must not match `my_store.py`.
        """
        name = self.collection_name(space)
        if name not in await self._db.list_collection_names():
            return []
        collection = self._db[name]
        wanted = rel_path.strip("/")

        cursor = await collection.find({"rel_path": wanted}).limit(limit).to_list()
        if cursor:
            return [_to_hit(document, score=0.0) for document in cursor]

        pattern = f"(^|/){re.escape(wanted)}$"
        documents = await collection.find({"rel_path": {"$regex": pattern}}).limit(limit).to_list()
        return [_to_hit(document, score=0.0) for document in documents]

    async def collection_names(self) -> list[str]:
        return sorted(await self._db.list_collection_names())

    async def scroll(
        self, space: EmbeddingSpace, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[tuple[str, dict[str, Any], dict[str, Any] | None]]:
        """Stream every document. What lets `reproject` build a truncated
        Matryoshka collection from vectors already paid for."""
        collection = self._db[self.collection_name(space)]
        cursor = collection.find({}).batch_size(batch_size)
        async for document in cursor:
            payload = {k: v for k, v in document.items() if k not in ("_id", DENSE_FIELD)}
            vectors = None
            if with_vectors and DENSE_FIELD in document:
                vectors = {DENSE_FIELD: _decode(document[DENSE_FIELD])}
            yield str(document["_id"]), payload, vectors

    async def drop_collection(self, space: EmbeddingSpace) -> None:
        name = self.collection_name(space)
        if name in await self._db.list_collection_names():
            await self._db.drop_collection(name)
            log.warning("store.collection_dropped", collection=name)
        self._ensured.discard(name)

    async def close(self) -> None:
        await self._client.close()


def build_document(
    chunk_id: str,
    vector: Sequence[float],
    payload: Mapping[str, object],
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    """One chunk as a BSON document, vector packed as binData.

    Module level and public because it is the whole storage decision in one
    place, and testable without a client: binData rather than an array of
    doubles is 2.4x smaller on this workspace's own numbers -- 6.15 KB against
    15.07 KB per document, 66 MB against 162 MB across 11,049 chunks -- which
    on a Free cluster capped at 512 MB including indexes is the difference
    between fitting and not fitting at all.
    """
    return {**payload, "_id": chunk_id, DENSE_FIELD: encode_vector(vector, dtype)}


def encode_vector(vector: Sequence[float], dtype: str = "float32") -> Binary:
    packed = _DTYPES[dtype]
    if packed is BinaryVectorDtype.INT8:
        # The quantised values themselves, not floats for Atlas to round.
        # Clamped because a component outside [-128, 127] raises rather than
        # saturating, and a value fractionally over 1.0 must not fail a batch
        # of 256 documents.
        return Binary.from_vector(
            [max(-128, min(127, round(value * 127))) for value in vector], packed
        )
    return Binary.from_vector(list(vector), packed)


def _to_hit(document: dict[str, Any], score: float | None = None) -> SearchHit:
    """Read a document back through the same function Qdrant's hits use.

    The payload is the contract; keeping one reader is what stops the two
    backends drifting into returning subtly different hits for the same chunk.
    """
    payload = {k: v for k, v in document.items() if k not in ("_id", DENSE_FIELD)}
    resolved = score if score is not None else float(document.get("score") or 0.0)
    return to_search_hit(str(document["_id"]), resolved, payload)


def _decode(value: Any) -> list[float]:
    if isinstance(value, Binary):
        return [float(v) for v in value.as_vector().data]
    return [float(v) for v in value]
