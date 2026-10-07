"""Qdrant implementation of the storage seam.

One collection per embedding space, because vectors from two different models
are not comparable — cosine distance between them is noise.

Both named vectors are declared at creation. Adding a named vector to a
populated Qdrant collection is not a simple migration, so deferring the sparse
half would cost a full re-embed later, which is exactly what the incremental
design exists to prevent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from qdrant_client import AsyncQdrantClient, models

from workspace_indexer.models import (
    Chunk,
    EmbeddingSpace,
    SearchFilters,
    SearchHit,
    SparseVec,
)
from workspace_indexer.obs.logging import get_logger, log_once
from workspace_indexer.storage.payload import INDEXED_FIELDS, to_payload, to_search_hit
from workspace_indexer.storage.query_spec import QuerySpec

log = get_logger("workspace_indexer.storage.qdrant")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


class QdrantStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        workspace: str,
        on_disk_payload: bool = True,
        upsert_batch_size: int = 256,
        payload_indexes: bool = True,
        location: str = "qdrant",
        owns_client: bool = True,
    ) -> None:
        self._client = client
        self._workspace = workspace
        # Passed in rather than read off the client's private attributes. The
        # client stores its target under a different name per mode, so sniffing
        # it works for embedded and silently returns nothing for server -- an
        # error message that omits the URL when the URL is the whole problem.
        self._location = location
        # False when the client was handed in and is shared with other stores.
        # Several workspaces served from one process share one client -- an
        # embedded Qdrant locks its storage folder to a single client, so a
        # second one is refused outright -- and the first store to close would
        # otherwise take the connection away from the rest.
        self._owns_client = owns_client
        self._on_disk_payload = on_disk_payload
        # Payload indexes are a no-op in embedded mode, and asking for them
        # emits a warning per field. Passed in rather than sniffed off the
        # client so the behaviour is explicit and testable.
        self._payload_indexes = payload_indexes
        self._batch_size = max(1, upsert_batch_size)
        self._ensured: set[str] = set()

    def collection_name(self, space: EmbeddingSpace) -> str:
        return f"{self._workspace}__{space.slug()}"

    def describe(self) -> str:
        """Where this store is, in words a human can act on."""
        return self._location

    async def ensure_collection(self, space: EmbeddingSpace) -> None:
        name = self.collection_name(space)
        if name in self._ensured:
            return

        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=space.dimensions, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        # Required for BM25. Without it Qdrant scores raw term
                        # frequency instead of inverse document frequency,
                        # which ranks badly and raises no error.
                        modifier=models.Modifier.IDF
                    )
                },
                # source_text lives in the payload, and would otherwise
                # dominate RAM.
                on_disk_payload=self._on_disk_payload,
            )
            log.info(
                "store.collection_created",
                collection=name,
                dimensions=space.dimensions,
                sparse_model=space.sparse_model,
            )

        await self._ensure_indexes(name)
        self._ensured.add(name)

    async def _ensure_indexes(self, name: str) -> None:
        if not self._payload_indexes:
            log_once(
                log,
                "store:no_payload_indexes",
                "store.payload_indexes_skipped",
                detail="embedded Qdrant ignores payload indexes; filters still work, "
                "but large filtered searches will scan",
            )
            return
        for field, schema in INDEXED_FIELDS.items():
            try:
                await self._client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema, wait=True
                )
            except Exception as exc:
                # Creating an index that already exists is the common case on
                # every run after the first, and is not a problem.
                log.debug("store.index_exists", collection=name, field=field, note=str(exc))

    async def upsert(
        self,
        space: EmbeddingSpace,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
    ) -> None:
        if not chunks:
            return
        if not (len(chunks) == len(dense) == len(sparse)):
            # A mismatch here would attach one chunk's vector to another's
            # payload, and nothing downstream could detect it.
            raise ValueError(
                f"upsert size mismatch: {len(chunks)} chunks, "
                f"{len(dense)} dense, {len(sparse)} sparse"
            )

        await self.upsert_points(
            space,
            [chunk.chunk_id for chunk in chunks],
            [list(vector) for vector in dense],
            list(sparse),
            [to_payload(chunk, space) for chunk in chunks],
        )

    async def upsert_points(
        self,
        space: EmbeddingSpace,
        ids: Sequence[str],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """Write points that did not come from a Chunk.

        Reprojection carries an existing payload across rather than rebuilding
        it, so it needs a way in that does not require re-chunking the file.
        """
        if not ids:
            return
        if not (len(ids) == len(dense) == len(sparse) == len(payloads)):
            raise ValueError(
                f"upsert size mismatch: {len(ids)} ids, {len(dense)} dense, "
                f"{len(sparse)} sparse, {len(payloads)} payloads"
            )

        await self.ensure_collection(space)
        name = self.collection_name(space)

        points = [
            models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR: list(dense_vector),
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                },
                payload=payload,
            )
            for point_id, dense_vector, sparse_vector, payload in zip(
                ids, dense, sparse, payloads, strict=True
            )
        ]

        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            await self._client.upsert(collection_name=name, points=batch, wait=True)
        log.info("store.upsert", collection=name, count=len(points))

    async def delete_by_ids(self, space: EmbeddingSpace, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        name = self.collection_name(space)
        await self._client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=list(chunk_ids)),
            wait=True,
        )
        log.info("store.delete", collection=name, count=len(chunk_ids), by="ids")

    async def delete_by_path(self, space: EmbeddingSpace, root_label: str, rel_path: str) -> None:
        """Deleting by path rather than by id is what makes file deletion and
        rename correct without first consulting the manifest."""
        name = self.collection_name(space)
        await self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="root_label", match=models.MatchValue(value=root_label)
                        ),
                        models.FieldCondition(
                            key="rel_path", match=models.MatchValue(value=rel_path)
                        ),
                    ]
                )
            ),
            wait=True,
        )
        log.info("store.delete", collection=name, by="path", rel_path=rel_path)

    async def search(
        self, space: EmbeddingSpace, query: QuerySpec, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        name = self.collection_name(space)
        condition = build_filter(filters)
        sparse_vector = (
            models.SparseVector(indices=query.sparse.indices, values=query.sparse.values)
            if query.sparse is not None
            else None
        )

        if query.fusion == "dense_only":
            if query.dense is None:
                return []
            response = await self._client.query_points(
                collection_name=name,
                query=query.dense,
                using=DENSE_VECTOR,
                query_filter=condition,
                limit=query.limit,
                with_payload=query.with_source,
            )
        elif query.fusion == "sparse_only":
            if sparse_vector is None:
                return []
            response = await self._client.query_points(
                collection_name=name,
                query=sparse_vector,
                using=SPARSE_VECTOR,
                query_filter=condition,
                limit=query.limit,
                with_payload=query.with_source,
            )
        else:
            prefetch: list[models.Prefetch] = []
            if query.dense is not None:
                prefetch.append(
                    models.Prefetch(
                        query=query.dense,
                        using=DENSE_VECTOR,
                        limit=query.prefetch_limit,
                        filter=condition,
                    )
                )
            if sparse_vector is not None:
                prefetch.append(
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR,
                        limit=query.prefetch_limit,
                        filter=condition,
                    )
                )
            if not prefetch:
                return []
            response = await self._client.query_points(
                collection_name=name,
                prefetch=prefetch,
                # RRF fuses on rank, not score, which is the only correct
                # choice here: cosine similarity and BM25 are not on a
                # comparable scale.
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=condition,
                limit=query.limit,
                with_payload=query.with_source,
            )

        hits = [
            to_search_hit(str(point.id), point.score, dict(point.payload or {}))
            for point in response.points
        ]
        log.info(
            "search.store",
            collection=name,
            fusion=query.fusion,
            returned=len(hits),
            filtered=condition is not None,
        )
        return hits

    async def describe_vectors(self, space: EmbeddingSpace) -> dict[str, list[str]]:
        """The named vectors a collection actually declares.

        Public because it answers a question worth asking from outside: a
        collection missing its sparse vector cannot do hybrid search, and
        adding one after the fact is not a simple migration.
        """
        info = await self._client.get_collection(self.collection_name(space))
        params = info.config.params
        dense = params.vectors
        dense_names = sorted(dense) if isinstance(dense, dict) else []
        sparse = params.sparse_vectors
        sparse_names = sorted(sparse) if isinstance(sparse, dict) else []
        return {"dense": dense_names, "sparse": sparse_names}

    async def count(self, space: EmbeddingSpace, filters: SearchFilters | None = None) -> int:
        name = self.collection_name(space)
        if not await self._client.collection_exists(name):
            return 0
        result = await self._client.count(
            collection_name=name, count_filter=build_filter(filters), exact=True
        )
        return result.count

    async def facet(self, space: EmbeddingSpace, key: str, limit: int = 32) -> dict[str, int]:
        """How many chunks carry each value of one payload field.

        One round trip for the whole distribution, rather than a count call per
        value. Values absent from the collection are absent from the result --
        the caller decides whether a missing key means zero or means unknown,
        because for the taxonomy those are very different answers.
        """
        name = self.collection_name(space)
        if not await self._client.collection_exists(name):
            return {}
        response = await self._client.facet(collection_name=name, key=key, limit=limit, exact=True)
        return {str(hit.value): hit.count for hit in response.hits}

    async def sample_paths(
        self, space: EmbeddingSpace, filters: SearchFilters | None = None, limit: int = 3
    ) -> list[str]:
        """A few distinct file paths matching a filter.

        For the taxonomy: a real path from this workspace calibrates a model
        far better than any prose definition of a category. Scrolls in pages
        because chunks cluster by file -- the first page can easily be twenty
        chunks of one document.
        """
        name = self.collection_name(space)
        if not await self._client.collection_exists(name):
            return []
        condition = build_filter(filters)
        found: list[str] = []
        offset: Any = None
        while len(found) < limit:
            points, offset = await self._client.scroll(
                collection_name=name,
                scroll_filter=condition,
                limit=64,
                offset=offset,
                with_payload=["rel_path"],
                with_vectors=False,
            )
            for point in points:
                path = str((point.payload or {}).get("rel_path", ""))
                if path and path not in found:
                    found.append(path)
                    if len(found) == limit:
                        break
            if offset is None:
                break
        return found

    async def chunks_for_path(
        self, space: EmbeddingSpace, rel_path: str, limit: int = 50
    ) -> list[SearchHit]:
        """Every indexed chunk of one file.

        Exact match first, because the caller almost always got `rel_path` from
        a search result and it is the payload value verbatim. A model that
        typed the path itself tends to give a suffix -- `state/manifest.py`
        for `workspace-indexer/src/.../state/manifest.py` -- so a miss falls
        back to scanning the path column and matching on a segment boundary.
        That scan reads one small keyword field, never a vector or a body, and
        the alternative is telling an agent its own file does not exist.
        """
        name = self.collection_name(space)
        if not await self._client.collection_exists(name):
            return []

        wanted = rel_path.strip("/")
        points, _ = await self._client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="rel_path", match=models.MatchValue(value=wanted))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            return [to_search_hit(str(p.id), 0.0, dict(p.payload or {})) for p in points]

        matched = await self._paths_ending_with(name, wanted)
        if not matched:
            return []
        points, _ = await self._client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="rel_path", match=models.MatchAny(any=matched))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [to_search_hit(str(p.id), 0.0, dict(p.payload or {})) for p in points]

    async def _paths_ending_with(self, name: str, suffix: str, max_pages: int = 40) -> list[str]:
        """Distinct rel_paths ending at a segment boundary of `suffix`.

        Bounded so a pathological collection cannot turn one tool call into an
        unbounded scan; `max_pages` of 4096 covers a workspace far larger than
        anything this is pointed at, and stopping early only costs a fallback,
        never a wrong answer.
        """
        needle = "/" + suffix
        found: set[str] = set()
        offset: Any = None
        for _ in range(max_pages):
            points, offset = await self._client.scroll(
                collection_name=name,
                limit=4096,
                offset=offset,
                with_payload=["rel_path"],
                with_vectors=False,
            )
            for point in points:
                path = str((point.payload or {}).get("rel_path", ""))
                if path.endswith(needle):
                    found.add(path)
            if offset is None:
                break
        return sorted(found)

    async def collection_names(self) -> list[str]:
        response = await self._client.get_collections()
        return sorted(c.name for c in response.collections)

    async def scroll(
        self, space: EmbeddingSpace, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[tuple[str, dict[str, Any], dict[str, Any] | None]]:
        """Stream every point. This is what lets `reproject` build a truncated
        Matryoshka collection from vectors we already paid for."""
        name = self.collection_name(space)
        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for point in points:
                vectors = point.vector if with_vectors else None
                yield str(point.id), dict(point.payload or {}), _as_vector_map(vectors)
            if offset is None:
                break

    async def drop_collection(self, space: EmbeddingSpace) -> None:
        name = self.collection_name(space)
        if await self._client.collection_exists(name):
            await self._client.delete_collection(collection_name=name)
            log.warning("store.collection_dropped", collection=name)
        self._ensured.discard(name)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _as_vector_map(vectors: object) -> dict[str, Any] | None:
    """Qdrant returns named vectors as a mapping; a single-vector collection
    returns a bare list, which we do not use."""
    if not isinstance(vectors, dict):
        return None
    # isinstance narrows `object` only to dict[Unknown, Unknown]; the client's
    # own annotation is the authority for the key and value types.
    return {str(key): value for key, value in cast("dict[str, Any]", vectors).items()}


def build_filter(filters: SearchFilters | None) -> models.Filter | None:
    """Filters run inside the search, never after it.

    Post-filtering a returned page would silently shrink the result set: ask
    for 10 hits in one repo and get 3 because the other 7 were elsewhere.
    """
    if filters is None or filters.is_empty():
        return None

    exact = {
        "root_label": filters.root_label,
        "unit": filters.unit,
        "repo_name": filters.repo_name,
        "language": filters.language,
        "symbol_kind": filters.symbol_kind,
        "kind": filters.kind.value if filters.kind else None,
        "doc_type": filters.doc_type.value if filters.doc_type else None,
    }
    must: list[models.Condition] = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in exact.items()
        if value is not None
    ]
    if filters.doc_types:
        # MatchAny rather than several MatchValue conditions: `must` is an AND,
        # so one condition per type would ask for a chunk that is somehow both
        # normative and design, and match nothing at all.
        must.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=[t.value for t in filters.doc_types]),
            )
        )

    if filters.path_prefix:
        prefix = filters.path_prefix.strip("/")
        must.append(models.FieldCondition(key="ancestors", match=models.MatchValue(value=prefix)))

    # must_not rather than a positive list: a caller saying "not tests" should
    # not have to enumerate every type it does want, and a type added to the
    # taxonomy later is then included by default rather than silently dropped.
    must_not: list[models.Condition] = [
        models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type.value))
        for doc_type in filters.exclude_doc_types
    ]

    if not must and not must_not:
        return None
    return models.Filter(must=must, must_not=must_not or None)
