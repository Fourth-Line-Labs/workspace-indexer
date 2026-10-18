"""Backend selection and credentials, from .env.

Kept separate from workspace.yaml on purpose: this half holds secrets and is
never committed, and it does not need to be hot-reloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from workspace_indexer.config.workspace_config import WorkspaceConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- dense embeddings ----
    embedding_model: str = "voyageai:voyage-code-4"
    embedding_dimensions: int = 2048
    embedding_quantization: Literal["float32", "int8", "binary"] = "float32"
    embedding_batch_size: int = 64
    embedding_max_concurrency: int = 4
    # Providers cap total tokens per request, not just document count: 64
    # chunks that each happen to be huge fails the whole batch.
    embedding_max_batch_tokens: int = 100_000
    voyage_api_key: str | None = None
    # What a million input tokens costs, used only when the provider does not
    # report a price. voyage-code-4 is $0.12/M at the time of writing, and
    # genai-prices has no entry for it -- so without this every run records
    # $0.0000, which reads as free rather than as unknown.
    #
    # A number in a config file goes stale silently, so anything priced this
    # way is reported as an estimate, never as a cost.
    embedding_price_per_mtok: float | None = None
    # Size of the provider's free allowance, if it has one. Used by `status` to
    # show how much of it this manifest has consumed. An approximation by
    # construction: the allowance belongs to the account and is drawn down by
    # everything using the key, not only by this index.
    embedding_free_tier_tokens: int | None = None

    # ---- evaluation ----
    # Colon-separated absolute paths whose contents MAY appear in a recorded
    # eval run. An allowlist: anything outside it is replaced by a digest
    # before the record is written. `evals/` is committed to a public
    # repository and the corpus is a whole workspace, so the default has to be
    # "do not publish". The denylist this replaced leaked, because a denylist
    # can only hide what someone remembered to name. Empty disables redaction
    # entirely, and the CLI says so on every recorded run.
    #
    # Deliberately not part of `config_hash`: it changes what is written down,
    # never what was measured.
    eval_public_roots: str = ""

    # ---- sparse ----
    sparse_model: str = "Qdrant/bm25"

    # ---- reranking ----
    rerank_enabled: bool = True
    rerank_model: str = "voyageai:rerank-2.5-lite"

    # ---- vector store ----
    vector_store: Literal["qdrant", "mongodb"] = "qdrant"
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_on_disk_payload: bool = True

    # ---- mongodb (only read when vector_store is mongodb) ----
    # The whole credential, so `repr=False`: a connection string carries the
    # password inline, and Settings gets logged and printed.
    mongodb_connection_string: str | None = Field(default=None, repr=False)
    mongodb_database: str = "workspace_indexer"
    # How the dense vector is packed into BSON. `float32` is binData at 4 bytes
    # per dimension -- 2.4x smaller than an array of doubles, measured. `int8`
    # is 4x smaller again and costs some recall; reach for it only when a Free
    # cluster's 512 MB is the binding constraint.
    mongodb_vector_dtype: Literal["float32", "int8"] = "float32"

    # ---- state ----
    state_db: Path = Path("./data/manifest.sqlite3")

    # ---- logging overrides (workspace.yaml wins unless these are set) ----
    log_level: str | None = None
    logfire_enabled: bool | None = None
    logfire_send_to_cloud: bool | None = None
    logfire_token: str | None = Field(default=None, repr=False)

    def export_credentials(self) -> list[str]:
        """Push API keys from .env into the process environment.

        pydantic-settings loads .env into this object, but provider SDKs --
        pydantic-ai's VoyageAI provider, and every other one -- read their
        credentials from os.environ. Without this bridge the key is loaded and
        then invisible to the thing that needs it, and the failure surfaces as
        "set the VOYAGE_API_KEY environment variable" while it is sitting right
        there in .env.

        A real environment variable always wins: the shell is more specific
        than a file, and overriding it would make `VOYAGE_API_KEY=x command`
        silently do nothing.

        Returns the names exported, for logging. Never the values.
        """
        exported: list[str] = []
        for field in type(self).model_fields:
            if not (field.endswith("_api_key") or field.endswith("_token")):
                continue
            value = getattr(self, field, None)
            if not value:
                continue
            name = field.upper()
            if os.environ.get(name):
                continue
            os.environ[name] = str(value)
            exported.append(name)
        return exported

    def config_hash(self, config: WorkspaceConfig) -> str:
        """Fingerprint of everything that affects index *content*.

        Credentials and transport settings are excluded — rotating an API key
        should not look like a configuration change that invalidates an index.

        `config` is required, not optional. This used to hash the embedding
        settings alone while claiming to cover "everything that affects index
        content", which left out the largest thing of all: *what was indexed*.
        Two runs over entirely different corpora produced the same hash, so
        `comparable_to` returned True and the delta between them was reported
        as though it meant something. An optional parameter would have let that
        back in through the default.
        """
        material = {
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_quantization": self.embedding_quantization,
            "sparse_model": self.sparse_model,
            # Resolved, so `~/src` and `/home/me/src` are one corpus rather
            # than two, and sorted so root order is not a change.
            "roots": sorted(
                f"{root.resolved_label}:{root.path.expanduser().resolve()}"
                for root in config.workspace.roots
            ),
            # A file excluded is a file not indexed, which moves recall exactly
            # as adding a root does.
            "excludes": sorted(config.all_excludes),
            "respect_gitignore": config.index.respect_gitignore,
            "max_file_bytes": config.index.max_file_bytes,
        }
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
