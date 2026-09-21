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
    from pathlib import Path as P

    tree = ast.parse(P("src/workspace_indexer/cli.py").read_text(encoding="utf-8"))
    missing: list[str] = []
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
            missing.append(node.name)

    assert not missing, f"commands taking --config but not --workspace: {missing}"


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
