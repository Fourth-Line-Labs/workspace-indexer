"""Several isolated indexes from one config file.

The failure this design exists to prevent is silent: the manifest keys every
table on `(root_label, rel_path)` and has no workspace column, so two
workspaces sharing one database collide on any root label they happen to share
-- which under `recurse_into_children` means any two trees each holding a
`docs/` or `src/`. Not an error. Wrong files and wrong chunks.

So the assertions here are mostly about separation being real and about the
single-workspace path being untouched, since every config written before this
existed uses it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from workspace_indexer.config import EmbeddingSection, Settings, WorkspaceChoiceError
from workspace_indexer.config import WorkspaceConfig as Config


def _raw(*names: str, state_dir: str | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "workspaces": [{"name": n, "roots": [{"path": f"/tmp/{n}"}]} for n in names]
    }
    if state_dir is not None:
        body["state_dir"] = state_dir
    return {**body, **extra}


# ---- the singular spelling keeps working -----------------------------------


def test_the_singular_spelling_is_one_workspace() -> None:
    """Every config written before this existed uses `workspace:`. If it needed
    editing, this change would break every deployment at once."""
    config = Config.model_validate({"workspace": {"name": "solo", "roots": [{"path": "/tmp"}]}})

    assert config.workspace_names == ["solo"]
    assert config.workspace.name == "solo"


def test_a_single_workspace_needs_no_state_dir() -> None:
    """It keeps using STATE_DB, so no existing manifest moves."""
    config = Config.model_validate({"workspace": {"name": "solo", "roots": [{"path": "/tmp"}]}})
    assert config.manifest_path(Path("./data/manifest.sqlite3")) == Path("./data/manifest.sqlite3")


def test_both_spellings_at_once_is_refused() -> None:
    """A config holding two answers to "what do I index" has no right reading,
    and guessing one would index something nobody asked for."""
    with pytest.raises(ValidationError, match="not both"):
        Config.model_validate(
            {
                "workspace": {"name": "a", "roots": [{"path": "/tmp"}]},
                "workspaces": [{"name": "b", "roots": [{"path": "/tmp"}]}],
            }
        )


# ---- separation is real ----------------------------------------------------


def test_several_workspaces_require_a_state_dir() -> None:
    """STATE_DB names one file. Letting several workspaces share it is the
    collision this design exists to prevent, so it is refused at config load
    rather than discovered as wrong search results."""
    with pytest.raises(ValidationError, match="state_dir"):
        Config.model_validate(_raw("alpha", "beta"))


def test_each_workspace_gets_its_own_manifest() -> None:
    config = Config.model_validate(_raw("alpha", "beta", state_dir="/var/idx"))

    alpha = config.select("alpha").manifest_path(Path("/unused.sqlite3"))
    beta = config.select("beta").manifest_path(Path("/unused.sqlite3"))

    assert alpha != beta
    assert alpha.parent == beta.parent == Path("/var/idx")


def test_identically_labelled_roots_in_two_workspaces_do_not_share_a_manifest() -> None:
    """The corruption case, asserted directly.

    Two trees each containing `src` produce the same root label, and the
    manifest keys on `(root_label, rel_path)` with nothing to tell the
    workspaces apart. Separate files are what stops one workspace's chunks
    being attributed to the other's files.
    """
    config = Config.model_validate(
        {
            "state_dir": "/var/idx",
            "workspaces": [
                {"name": "alpha", "roots": [{"path": "/a/src"}]},
                {"name": "beta", "roots": [{"path": "/b/src"}]},
            ],
        }
    )

    labels = {w.roots[0].resolved_label for w in config.workspaces}
    paths = {config.select(w.name).manifest_path(Path("/x")) for w in config.workspaces}

    assert labels == {"src"}, "the collision this guards against is not being reproduced"
    assert len(paths) == 2


def test_two_workspaces_cannot_share_a_name() -> None:
    """The name keys both the collection and the manifest file, so a duplicate
    is two workspaces writing over each other."""
    with pytest.raises(ValidationError, match="duplicate workspace names"):
        Config.model_validate(_raw("same", "same", state_dir="/var/idx"))


# ---- refusing to guess -----------------------------------------------------


def test_selecting_nothing_where_there_is_a_choice_raises_and_lists_them() -> None:
    config = Config.model_validate(_raw("alpha", "beta", state_dir="/var/idx"))

    with pytest.raises(WorkspaceChoiceError) as caught:
        config.select()

    assert "alpha" in str(caught.value)
    assert "beta" in str(caught.value)


def test_an_unknown_workspace_names_the_configured_ones() -> None:
    """The reply to "which one?" and to "that one does not exist" is the same
    list, so they are one error."""
    config = Config.model_validate(_raw("alpha", "beta", state_dir="/var/idx"))

    with pytest.raises(WorkspaceChoiceError, match="alpha, beta"):
        config.select("gamma")


def test_the_workspace_property_refuses_to_pick_one() -> None:
    """Everything downstream of `select` reads this. Answering from whichever
    was listed first would serve one client's code to a question about
    another's, silently."""
    config = Config.model_validate(_raw("alpha", "beta", state_dir="/var/idx"))

    with pytest.raises(WorkspaceChoiceError):
        _ = config.workspace


def test_selecting_by_name_is_unnecessary_for_a_lone_workspace() -> None:
    config = Config.model_validate({"workspace": {"name": "solo", "roots": [{"path": "/tmp"}]}})
    assert config.select().workspace.name == "solo"


# ---- overrides are folded in, not carried ----------------------------------


def test_selecting_folds_an_eval_override_into_the_shared_section() -> None:
    """Nothing downstream should have to know an override was possible: the
    indexer and the walker receive what looks like the single-workspace config
    they have always received."""
    config = Config.model_validate(
        {
            "state_dir": "/var/idx",
            "eval": {"dataset": "/shared/eval.yaml"},
            "workspaces": [
                {"name": "alpha", "roots": [{"path": "/a"}], "eval": {"dataset": "/a/eval.yaml"}},
                {"name": "beta", "roots": [{"path": "/b"}]},
            ],
        }
    )

    assert config.select("alpha").eval.dataset == Path("/a/eval.yaml")
    assert config.select("beta").eval.dataset == Path("/shared/eval.yaml")


def test_every_workspace_dataset_is_excluded_from_every_index() -> None:
    """A dataset is a perfect match for its own queries, so indexing one
    corrupts the measurement taken from it. Excluding all of them costs nothing
    and covers a dataset that happens to sit inside another workspace's tree."""
    config = Config.model_validate(
        {
            "state_dir": "/var/idx",
            "eval": {"dataset": "/shared/eval.yaml"},
            "workspaces": [
                {"name": "alpha", "roots": [{"path": "/a"}], "eval": {"dataset": "/a/eval.yaml"}},
                {"name": "beta", "roots": [{"path": "/b"}], "eval": {"dataset": "/b/eval.yaml"}},
            ],
        }
    )

    excluded = config.select("alpha").excluded_paths

    assert Path("/a/eval.yaml") in excluded
    assert Path("/b/eval.yaml") in excluded


# ---- embedding per workspace -----------------------------------------------


def test_an_embedding_override_lands_on_settings() -> None:
    """Carried on Settings rather than separately so everything derived from it
    moves together -- the space, the backend, the price."""
    settings = Settings(embedding_model="voyageai:voyage-code-4", embedding_dimensions=2048)

    overridden = EmbeddingSection(
        model="fastembed:BAAI/bge-small-en-v1.5", dimensions=384
    ).applied_to(settings)

    assert overridden.embedding_model == "fastembed:BAAI/bge-small-en-v1.5"
    assert overridden.embedding_dimensions == 384


def test_an_override_changes_the_configuration_hash() -> None:
    """`eval --compare` only diffs runs whose hash matches. Without this, two
    workspaces on different models would produce runs claiming a comparability
    they do not have."""
    config = Config.model_validate({"workspace": {"name": "solo", "roots": [{"path": "/tmp"}]}})
    settings = Settings(embedding_model="voyageai:voyage-code-4", embedding_dimensions=2048)

    overridden = EmbeddingSection(dimensions=384).applied_to(settings)

    assert settings.config_hash(config) != overridden.config_hash(config)


def test_an_override_leaves_everything_it_does_not_name_alone() -> None:
    """Credentials in particular: the API key lives in .env and an embedding
    block in a committed file must not be able to disturb it."""
    settings = Settings(voyage_api_key="secret", state_db=Path("/data/m.sqlite3"))

    overridden = EmbeddingSection(dimensions=384).applied_to(settings)

    assert overridden.voyage_api_key == "secret"
    assert overridden.state_db == Path("/data/m.sqlite3")


def test_a_workspace_that_overrides_nothing_is_unchanged() -> None:
    """The ordinary case. An empty block must not quietly rewrite settings with
    its own defaults."""
    settings = Settings(embedding_model="voyageai:voyage-code-4", embedding_dimensions=2048)

    assert EmbeddingSection().applied_to(settings) is settings


# ---- the CLI surface -------------------------------------------------------


def test_every_command_that_takes_a_config_also_takes_a_workspace() -> None:
    """A command that reads a config but cannot be pointed at a workspace is
    unusable the moment a second one is configured -- and it fails by raising
    from deep inside rather than by saying which flag is missing.

    Read out of the source rather than listed here, so adding a command fails
    this until it is threaded through, not until someone remembers a list.
    """
    import ast

    # Anchored to this file rather than the working directory, like every other
    # test that reads repo source. A CWD-relative path passes or fails on where
    # pytest was invoked from.
    source = Path(__file__).resolve().parents[1] / "src" / "workspace_indexer" / "cli.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    declared: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "command"
            for d in node.decorator_list
        ):
            continue
        names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
        if "config" in names and "workspace" not in names:
            declared.append(node.name)

    assert not declared, f"commands taking --config but not --workspace: {declared}"


def test_the_workspace_flag_is_passed_on_and_not_merely_accepted() -> None:
    """Declaring the parameter is not using it.

    A command that accepts `--workspace` and quietly ignores it is the failure
    nearest the flag's promise: it would answer from the wrong workspace while
    looking like it honoured the request. So every command body must mention
    the name somewhere, not just its signature.
    """
    import ast

    source = Path(__file__).resolve().parents[1] / "src" / "workspace_indexer" / "cli.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    unused: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "command"
            for d in node.decorator_list
        ):
            continue
        names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
        if "workspace" not in names:
            continue
        used = any(
            isinstance(inner, ast.Name) and inner.id == "workspace"
            for statement in node.body
            for inner in ast.walk(statement)
        )
        if not used:
            unused.append(node.name)

    assert not unused, f"commands accepting --workspace without using it: {unused}"


# ---- the separation holds through a real indexing run ----------------------


async def test_two_workspaces_sharing_a_root_label_keep_their_files_apart(
    tmp_path: Path,
) -> None:
    """The corruption case, driven through the pipeline rather than argued
    about.

    Both roots are named `src`, so both produce the root label `src` -- and the
    manifest keys every table on `(root_label, rel_path)` with nothing to tell
    the workspaces apart. Each file is also named the same. If the two shared a
    database, one workspace's chunks would be attributed to the other's file
    and nothing would report an error.
    """
    import sqlite3

    from qdrant_client import AsyncQdrantClient

    from tests.indexer_harness import SPACE, Harness
    from workspace_indexer.state import Manifest
    from workspace_indexer.storage.qdrant_store import QdrantStore

    for name in ("alpha", "beta"):
        source = tmp_path / name / "src"
        source.mkdir(parents=True)
        (source / "same_name.py").write_text(f"def {name}_only():\n    return '{name}'\n")

    state = tmp_path / "state"
    state.mkdir()
    config = Config.model_validate(
        {
            "state_dir": str(state),
            "workspaces": [
                {"name": f"ws-{n}", "roots": [{"path": str(tmp_path / n / "src")}]}
                for n in ("alpha", "beta")
            ],
        }
    )
    assert {w.roots[0].resolved_label for w in config.workspaces} == {"src"}, (
        "the labels do not collide, so this proves nothing"
    )

    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    try:
        for name in ("alpha", "beta"):
            selected = config.select(f"ws-{name}")
            store = QdrantStore(client, workspace=selected.workspace.name, payload_indexes=False)
            with Manifest(selected.manifest_path(tmp_path / "unused.sqlite3")) as manifest:
                await Harness(selected, store, manifest, tmp_path).indexer(SPACE).run()
    finally:
        await client.close()

    contents: dict[str, list[str]] = {}
    for name in ("alpha", "beta"):
        with sqlite3.connect(state / f"ws-{name}.sqlite3") as database:
            rows = database.execute("SELECT root_label, rel_path FROM files").fetchall()
        contents[name] = [f"{r[0]}/{r[1]}" for r in rows]

    # Each workspace saw exactly one file, and they are indistinguishable by
    # the key the manifest uses -- which is the whole point.
    assert contents["alpha"] == contents["beta"] == ["src/same_name.py"]
    assert (state / "ws-alpha.sqlite3").exists()
    assert (state / "ws-beta.sqlite3").exists()


# ---- the fixes from review -------------------------------------------------


@pytest.mark.parametrize("name", ["../escape", "nested/deep", "", ".hidden", "a b"])
def test_a_workspace_name_that_would_escape_state_dir_is_refused(name: str) -> None:
    """The name becomes a filename, and `Manifest` creates parent directories,
    so an unchecked one succeeds silently in the wrong place rather than
    failing."""
    with pytest.raises(ValidationError):
        Config.model_validate({"workspace": {"name": name, "roots": [{"path": "/a"}]}})


def test_names_differing_only_in_case_are_refused() -> None:
    """On Windows and macOS they are one file -- so this is the
    two-workspaces-one-manifest collision `state_dir` exists to prevent,
    arriving through the name instead of the root label."""
    with pytest.raises(ValidationError, match="duplicate workspace names"):
        Config.model_validate(
            {
                "state_dir": "/var/idx",
                "workspaces": [
                    {"name": "Alpha", "roots": [{"path": "/a"}]},
                    {"name": "alpha", "roots": [{"path": "/b"}]},
                ],
            }
        )


def test_a_partial_eval_block_inherits_the_fields_it_does_not_name() -> None:
    """Every `EvalSection` field has a default, so replacing the section
    wholesale gives a workspace that only wanted different metrics the *class
    default* dataset -- evaluating against a file nobody configured."""
    config = Config.model_validate(
        {
            "state_dir": "/var/idx",
            "eval": {"dataset": "/shared/eval.yaml"},
            "workspaces": [
                {
                    "name": "alpha",
                    "roots": [{"path": "/a"}],
                    "eval": {"metrics": ["recall@5"]},
                },
                {"name": "beta", "roots": [{"path": "/b"}]},
            ],
        }
    )

    alpha = config.select("alpha").eval

    assert alpha.metrics == ["recall@5"]
    assert alpha.dataset == Path("/shared/eval.yaml")


def test_an_embedding_model_without_its_dimensions_is_refused() -> None:
    """Nothing cross-validates the pair at runtime: the space would claim the
    inherited width while the backend returned another, and the mismatch
    surfaces at the first embed batch, after tokens have been spent."""
    with pytest.raises(ValidationError, match="dimensions"):
        EmbeddingSection(model="fastembed:BAAI/bge-small-en-v1.5")


def test_every_override_maps_onto_a_real_setting() -> None:
    """The keys are built by an `embedding_` convention with one name that
    breaks it, and `Settings` ignores extras -- so a key that stopped matching
    would be dropped by the round trip with no error and no effect."""
    settings = Settings()

    applied = EmbeddingSection(
        model="fastembed:BAAI/bge-small-en-v1.5",
        dimensions=384,
        quantization="int8",
        sparse_model="Qdrant/bm25",
        price_per_mtok=0.5,
    ).applied_to(settings)

    assert applied.embedding_model == "fastembed:BAAI/bge-small-en-v1.5"
    assert applied.embedding_dimensions == 384
    assert applied.embedding_quantization == "int8"
    assert applied.sparse_model == "Qdrant/bm25"
    assert applied.embedding_price_per_mtok == 0.5


def test_other_workspaces_stay_excluded_after_the_rerank_overrides_are_applied() -> None:
    """`with_rerank_overrides` rebuilds the config with `model_copy`, and the
    cross-workspace dataset exclusion rides on private state that copy happens
    to carry. If that ever changes, the exclusion dies silently and only on
    machines that set `RERANK_*` -- so it is pinned on the public seam.
    """
    from workspace_indexer.app_context import with_rerank_overrides

    config = Config.model_validate(
        {
            "state_dir": "/var/idx",
            "workspaces": [
                {"name": "alpha", "roots": [{"path": "/a"}], "eval": {"dataset": "/a/eval.yaml"}},
                {"name": "beta", "roots": [{"path": "/b"}], "eval": {"dataset": "/b/eval.yaml"}},
            ],
        }
    )

    rebuilt = with_rerank_overrides(
        config.select("alpha"), Settings(rerank_model="voyageai:rerank-2.5-lite")
    )

    assert Path("/b/eval.yaml") in rebuilt.excluded_paths


def test_an_embedding_override_does_not_discard_the_yaml_rerank_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` wins over workspace.yaml only for settings someone actually set.

    `EmbeddingSection.applied_to` rebuilds `Settings` from a full
    `model_dump()`, and pydantic takes `model_fields_set` from the input
    dict's keys -- so afterwards every field counts as explicitly provided.
    `with_rerank_overrides` gates on exactly that signal, which is the only
    thing telling "the default" from "someone typed the default". Read off the
    derived settings it overrides workspace.yaml every time, silently, because
    the result is re-validated.

    The workspace it hits hardest is the one the feature exists for: barred
    from a hosted API, embedding locally, reranking deliberately off -- and
    getting hosted reranking switched back on.
    """
    from workspace_indexer.app_context import AppContext

    monkeypatch.chdir(tmp_path)
    # The same list the registry tests clear. A narrower one lets a developer
    # shell redirect this: an exported VECTOR_STORE=mongodb makes
    # `build_vector_store` raise once chdir has moved off the repo `.env`, and
    # an exported SPARSE_MODEL changes what gets built -- either way a rerank
    # test quietly stops testing reranking.
    for key in (
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "SPARSE_MODEL",
        "VOYAGE_API_KEY",
        "VECTOR_STORE",
        "RERANK_ENABLED",
        "RERANK_MODEL",
        "QDRANT_MODE",
        "QDRANT_PATH",
        "STATE_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("STATE_DB", str(tmp_path / "manifest.sqlite3"))

    config = Config.model_validate(
        {
            "workspace": {
                "name": "walled",
                "roots": [{"path": str(tmp_path)}],
                "embedding": {"model": "voyageai:voyage-code-4", "dimensions": 256},
            },
            "search": {"rerank": {"enabled": False, "model": "local:some/model"}},
        }
    )

    context = AppContext.from_config(config)
    try:
        assert context.config.search.rerank.enabled is False
        assert context.config.search.rerank.model == "local:some/model"
        # The embedding override still has to have landed, or this passes for
        # the wrong reason -- by the override never being applied at all.
        assert context.settings.embedding_dimensions == 256
    finally:
        import asyncio

        asyncio.run(context.close())


def test_a_context_that_fails_to_build_closes_what_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest opens before the things that can fail.

    `build_reranker` rejecting an unknown provider happens after the sqlite
    handle exists, and the caller never receives the half-built context -- so
    the handle would be orphaned with nobody able to close it. Synchronous
    because the store half of the cleanup is skipped inside a running loop.
    """
    import workspace_indexer.app_context as app_context
    from workspace_indexer.app_context import AppContext

    monkeypatch.chdir(tmp_path)
    # The same ten keys as the sibling test above. `from_config` builds the
    # embedding service and applies the rerank overrides *before* the patched
    # `build_reranker` is reached, so an exported EMBEDDING_MODEL,
    # EMBEDDING_DIMENSIONS, VOYAGE_API_KEY or RERANK_MODEL can fail the
    # construction this test needs to get through -- surfacing a different
    # error before any manifest exists to assert on.
    for key in (
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "SPARSE_MODEL",
        "VOYAGE_API_KEY",
        "VECTOR_STORE",
        "RERANK_ENABLED",
        "RERANK_MODEL",
        "QDRANT_MODE",
        "QDRANT_PATH",
        "STATE_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("STATE_DB", str(tmp_path / "manifest.sqlite3"))

    opened: list[Any] = []
    real_manifest = app_context.Manifest

    def record(path: Path) -> Any:
        manifest = real_manifest(path)
        opened.append(manifest)
        return manifest

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unknown rerank provider 'nonesuch'")

    monkeypatch.setattr(app_context, "Manifest", record)
    monkeypatch.setattr(app_context, "build_reranker", refuse)

    config = Config.model_validate({"workspace": {"name": "w", "roots": [{"path": str(tmp_path)}]}})

    with pytest.raises(ValueError, match="unknown rerank provider"):
        AppContext.from_config(config)

    assert opened, "no manifest was opened, so this asserts nothing"
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].file_count()


def test_a_store_that_will_not_build_does_not_orphan_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_vector_store` is itself a thing that can fail -- mongodb with no
    connection string, an embedded Qdrant whose storage folder is already
    held. It used to run before the guarded window, so its failure left the
    sqlite handle opened a line earlier with nobody able to close it.
    """
    import workspace_indexer.app_context as app_context
    from workspace_indexer.app_context import AppContext

    monkeypatch.chdir(tmp_path)
    for key in (
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "SPARSE_MODEL",
        "VOYAGE_API_KEY",
        "VECTOR_STORE",
        "RERANK_ENABLED",
        "RERANK_MODEL",
        "QDRANT_MODE",
        "QDRANT_PATH",
        "STATE_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("STATE_DB", str(tmp_path / "manifest.sqlite3"))

    opened: list[Any] = []
    real_manifest = app_context.Manifest

    def record(path: Path) -> Any:
        manifest = real_manifest(path)
        opened.append(manifest)
        return manifest

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("VECTOR_STORE=mongodb needs MONGODB_CONNECTION_STRING")

    monkeypatch.setattr(app_context, "Manifest", record)
    monkeypatch.setattr(app_context, "build_vector_store", refuse)

    config = Config.model_validate({"workspace": {"name": "w", "roots": [{"path": str(tmp_path)}]}})

    with pytest.raises(ValueError, match="MONGODB_CONNECTION_STRING"):
        AppContext.from_config(config)

    assert opened, "no manifest was opened, so this asserts nothing"
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].file_count()
