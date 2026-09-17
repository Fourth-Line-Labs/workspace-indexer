"""Every setting must be read by something.

A setting that nothing reads is worse than a setting that does not exist. It is
documented, it appears in `.env.example`, someone sets it, and nothing happens
-- silently, with no error and no log line, which is the failure mode this
project keeps running into from a new direction.

It had already happened twice when this was written. `RERANK_ENABLED` and
`RERANK_MODEL` were declared in `Settings`, listed in the reference and copied
into `.env.example`, and read by nothing at all: the reranker is built from
`config.search.rerank`, so the two settings sat next to it doing nothing. I
found them by setting `RERANK_MODEL=database:rerank-2.5-lite` to benchmark
server-side reranking, watching a client-side Voyage reranker run instead, and
nearly reporting the resulting numbers as a measurement of something else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from workspace_indexer.config import Settings

SRC = Path(__file__).resolve().parents[1] / "src" / "workspace_indexer"

# Settings that are deliberately inert, each with the reason. A name here is a
# claim that the reference says so too, which the second test checks -- an
# undocumented dead setting and a documented one are different problems and
# only the first is a bug.
DELIBERATELY_INERT: dict[str, str] = {
    "embedding_quantization": (
        "accepted and unused since iteration 1; int8/binary storage is not "
        "implemented and the reference says so"
    ),
    "logfire_token": (
        "read from the environment by the logfire SDK rather than by us; "
        "declared here so it is documented and so pydantic does not reject it"
    ),
}


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Away from the repository's own `.env`.

    `Settings()` reads `.env` from the working directory, so without this a
    test asserting "nothing was set" asserts something about the developer's
    machine instead -- and passes or fails depending on whose laptop it is.
    Same pattern as `test_store_factory`.
    """
    monkeypatch.chdir(tmp_path)


def _source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC.rglob("*.py")
        # settings.py declares them; reading them there would prove nothing.
        if path.name != "settings.py"
    )


def test_every_setting_is_read_somewhere() -> None:
    source = _source()
    dead = [
        name
        for name in Settings.model_fields
        if name not in DELIBERATELY_INERT and not re.search(rf"\.{name}\b", source)
    ]
    assert not dead, (
        "these settings are declared and documented but nothing reads them, so "
        "setting them does nothing at all:\n  " + "\n  ".join(dead)
    )


def test_a_deliberately_inert_setting_says_so_in_the_reference() -> None:
    """An inert setting is acceptable; an inert setting documented as working
    is a trap. The reference has to admit it."""
    reference = (SRC.parents[1] / "docs" / "reference.md").read_text(encoding="utf-8")
    undisclosed = [
        name
        for name in DELIBERATELY_INERT
        if not re.search(
            rf"`{name.upper()}`[^\n]*(unused|not read|SDK|environment)", reference, re.IGNORECASE
        )
    ]
    assert not undisclosed, (
        f"these settings do nothing, and the reference does not say so: {undisclosed}"
    )


def test_the_inert_list_does_not_outlive_the_settings() -> None:
    """The allowlist is a list too, and drifts like any other."""
    stale = [name for name in DELIBERATELY_INERT if name not in Settings.model_fields]
    assert not stale, f"DELIBERATELY_INERT names settings that no longer exist: {stale}"


def test_an_inert_setting_that_starts_working_leaves_the_list() -> None:
    """Otherwise the list slowly becomes a description of nothing, and stops
    being read by anyone deciding whether to trust a setting."""
    source = _source()
    now_wired = [name for name in DELIBERATELY_INERT if re.search(rf"\.{name}\b", source)]
    assert not now_wired, (
        "these settings are now read somewhere and should be removed from "
        f"DELIBERATELY_INERT: {now_wired}"
    )


def test_the_rerank_model_env_override_reaches_the_reranker(tmp_path: Path) -> None:
    """The bug this file was written for.

    `RERANK_MODEL` was declared, documented and read by nothing, so setting it
    to `database:rerank-2.5-lite` ran a client-side Voyage reranker instead --
    and produced numbers that looked like a successful measurement of
    server-side reranking.
    """
    from workspace_indexer.app_context import with_rerank_overrides
    from workspace_indexer.config import WorkspaceConfig

    config = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": str(tmp_path), "label": "main"}]}}
    )
    assert config.search.rerank.model == "voyageai:rerank-2.5-lite"

    settings = Settings(rerank_model="local:BAAI/bge-reranker-base")
    overridden = with_rerank_overrides(config, settings)
    assert overridden.search.rerank.model == "local:BAAI/bge-reranker-base"
    assert overridden.search.rerank.provider == "local"


def test_an_override_is_validated_rather_than_copied_in(tmp_path: Path) -> None:
    """`model_copy(update=...)` does not run field validators, so every check on
    `RerankConfig.model` was dead on the one path people actually use: .env.

    Both cases below reached the reranker factory instead, where a missing
    provider prefix and the retired `database:` provider alike came back as
    "unknown rerank provider" -- naming neither the real requirement nor the
    way out.
    """
    from workspace_indexer.app_context import with_rerank_overrides
    from workspace_indexer.config import WorkspaceConfig

    config = WorkspaceConfig.model_validate(
        {"workspace": {"name": "w", "roots": [{"path": str(tmp_path), "label": "main"}]}}
    )

    with pytest.raises(ValidationError, match="must be `provider:model`"):
        with_rerank_overrides(config, Settings(rerank_model="rerank-2.5-lite"))

    with pytest.raises(ValidationError, match="8.3"):
        with_rerank_overrides(config, Settings(rerank_model="database:rerank-2.5-lite"))


def test_an_unset_override_leaves_workspace_yaml_alone(tmp_path: Path) -> None:
    """`rerank_enabled` and `rerank_model` have non-None defaults, so applying
    them unconditionally would override a workspace.yaml that configured
    reranking deliberately. Only what was actually set may win."""
    from workspace_indexer.app_context import with_rerank_overrides
    from workspace_indexer.config import WorkspaceConfig

    config = WorkspaceConfig.model_validate(
        {
            "workspace": {"name": "w", "roots": [{"path": str(tmp_path), "label": "main"}]},
            "search": {"rerank": {"model": "fastembed:some/model", "enabled": False}},
        }
    )
    unchanged = with_rerank_overrides(config, Settings())

    assert unchanged.search.rerank.model == "fastembed:some/model"
    assert unchanged.search.rerank.enabled is False
