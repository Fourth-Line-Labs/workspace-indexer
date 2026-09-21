"""Configuration models.

workspace.yaml is *what to index* and is safe to commit; .env is *how to
index* and holds credentials. Only the former needs hot-reloading, which keeps
the watcher simple.

One class per module, re-exported here for readable call sites.
"""

from workspace_indexer.config.chunking_section import ChunkingSection
from workspace_indexer.config.code_chunking import CodeChunking
from workspace_indexer.config.config_error import ConfigError
from workspace_indexer.config.embedding_section import EmbeddingSection
from workspace_indexer.config.eval_section import EvalSection
from workspace_indexer.config.excludes import HARDCODED_EXCLUDES
from workspace_indexer.config.file_log_config import FileLogConfig
from workspace_indexer.config.graph_section import DEFAULT_HTTP_CLIENTS, GraphSection
from workspace_indexer.config.index_section import IndexSection
from workspace_indexer.config.loader import DEFAULT_CONFIG_PATH, load_workspace_config
from workspace_indexer.config.logfire_config import LogfireConfig
from workspace_indexer.config.logging_config import LoggingConfig
from workspace_indexer.config.markdown_chunking import MarkdownChunking
from workspace_indexer.config.opaque_chunking import OpaqueChunking
from workspace_indexer.config.rerank_config import RerankConfig
from workspace_indexer.config.root_config import RootConfig
from workspace_indexer.config.search_section import SearchSection
from workspace_indexer.config.settings import Settings
from workspace_indexer.config.strict import Strict
from workspace_indexer.config.text_chunking import TextChunking
from workspace_indexer.config.watch_mode import WatchMode
from workspace_indexer.config.watch_section import WatchSection
from workspace_indexer.config.workspace_choice_error import WorkspaceChoiceError
from workspace_indexer.config.workspace_config import WorkspaceConfig
from workspace_indexer.config.workspace_section import WorkspaceSection

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HARDCODED_EXCLUDES",
    "ChunkingSection",
    "CodeChunking",
    "ConfigError",
    "DEFAULT_HTTP_CLIENTS",
    "EmbeddingSection",
    "EvalSection",
    "GraphSection",
    "FileLogConfig",
    "IndexSection",
    "LogfireConfig",
    "LoggingConfig",
    "MarkdownChunking",
    "OpaqueChunking",
    "RerankConfig",
    "RootConfig",
    "SearchSection",
    "WatchMode",
    "WatchSection",
    "Settings",
    "Strict",
    "TextChunking",
    "WorkspaceChoiceError",
    "WorkspaceConfig",
    "WorkspaceSection",
    "load_workspace_config",
]
