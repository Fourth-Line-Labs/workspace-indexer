"""Keeping paths that are nobody else's business out of a public repository.

`evals/` is committed, and an eval searches a whole workspace. Nothing here is
about retrieval quality: it is about what reaches git.

An allowlist, after a denylist version of this leaked a named employer and a
directory of job applications into pushed records -- it was configured with the
two client trees someone had thought of, and the verification scan looked for
those same two names, so it could not have found anything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.evaluation.eval_record import EvalRecord
from workspace_indexer.evaluation.eval_result import EvalResult
from workspace_indexer.evaluation.private_path_error import PrivatePathError
from workspace_indexer.evaluation.public_path_guard import ENV_VAR, PublicPathGuard


def _record(found: list[str]) -> EvalRecord:
    return EvalRecord(
        recorded_at="2026-10-18T00:00:00+00:00",
        label="t",
        config_hash="h",
        space_slug="n/a",
        embedding_model="n/a",
        dimensions=0,
        fusion="n/a",
        reranker="n/a",
        limit=10,
        recall_at_k=0.0,
        mrr_at_k=0.0,
        case_count=1,
        miss_count=0,
        results=[EvalResult(query="q", expected=["mine/a.py"], found=found)],
    )


def _found(guard: PublicPathGuard, paths: list[str]) -> list[str]:
    return guard.apply(_record(paths)).results[0].found


def test_a_path_outside_the_allowlist_is_replaced_by_a_stable_digest(tmp_path: Path) -> None:
    """Redaction rather than refusal: matches outside the allowlist happen on
    most runs, so a guard that rejected such a record would reject every one."""
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    first = _found(guard, ["theirs/secret.cs"])[0]
    second = _found(guard, ["theirs/secret.cs"])[0]

    assert "theirs" not in first and "secret" not in first
    assert first == second


def test_an_unlisted_tree_is_redacted_without_anyone_naming_it(tmp_path: Path) -> None:
    """The whole reason for the inversion. A denylist publishes what nobody
    thought to add to it; an allowlist redacts it."""
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    assert _found(guard, ["never-considered/cv.md"])[0].startswith("redacted/")


def test_a_listed_tree_is_untouched(tmp_path: Path) -> None:
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    assert _found(guard, ["mine/a.py"]) == ["mine/a.py"]


def test_a_sibling_sharing_a_name_prefix_is_not_treated_as_inside(tmp_path: Path) -> None:
    """`mine-notes` is not under `mine`. Matching on the raw string would
    publish it."""
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    assert _found(guard, ["mine-notes/readme.md"])[0].startswith("redacted/")


def test_two_redacted_files_stay_distinguishable(tmp_path: Path) -> None:
    """One placeholder for everything would make three hits on one file and
    three hits on three files read identically."""
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    assert len(set(_found(guard, ["x/a.cs", "x/b.cs"]))) == 2


def test_a_public_root_equal_to_the_workspace_root_publishes_everything(tmp_path: Path) -> None:
    """`relative_to` returns `Path('.')` here, whose string form matches no
    recorded path -- so the denylist version of this silently redacted nothing
    in its maximal configuration. The empty relative form is what makes the
    declaration mean what it says."""
    guard = PublicPathGuard([tmp_path], [tmp_path])

    assert guard.active
    assert _found(guard, ["anything/at/all.py"]) == ["anything/at/all.py"]


def test_a_directory_named_like_the_redaction_namespace_does_not_break_the_backstop(
    tmp_path: Path,
) -> None:
    """The backstop checks what was private *before* redaction. Re-testing the
    redacted output would report every redaction as a leak for anyone whose
    allowlist excludes `redacted/`, failing closed on an ordinary name and
    killing the measurement."""
    guard = PublicPathGuard([tmp_path / "mine"], [tmp_path])

    out = _found(guard, ["redacted/notes.md"])

    assert out[0].startswith("redacted/")
    assert out[0] != "redacted/notes.md"


def test_an_unresolved_spelling_still_matches(tmp_path: Path) -> None:
    """A symlinked root configured one way and declared the other would match
    nothing, and the failure mode of this control is to publish."""
    real = tmp_path / "real"
    (real / "mine").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)

    guard = PublicPathGuard([link / "mine"], [real])

    assert _found(guard, ["mine/a.py"]) == ["mine/a.py"]


def test_nothing_configured_means_nothing_redacted(tmp_path: Path) -> None:
    """A workspace with nothing to hide is not forced to enumerate itself. The
    CLI prints a warning on any recorded run in this state."""
    guard = PublicPathGuard([], [tmp_path])

    assert not guard.active
    assert _found(guard, ["anything.py"]) == ["anything.py"]


def test_a_public_root_outside_every_workspace_root_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It cannot match the root-relative form a run records, so it protects
    nothing -- and a silently useless privacy control looks exactly like a
    working one. Read off stdout rather than caplog: structlog renders there
    and never reaches the stdlib handler caplog installs."""
    PublicPathGuard([tmp_path / "elsewhere"], [tmp_path / "workspace"])

    assert "public_root_outside_workspace" in capsys.readouterr().out


def test_roots_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Read from the environment rather than workspace.yaml: the layout of a
    private tree should not enter a committed file."""
    monkeypatch.setenv(ENV_VAR, f"{tmp_path / 'a'}:{tmp_path / 'b'}")

    guard = PublicPathGuard.from_env([tmp_path])

    assert guard.is_public("a/x.py")
    assert guard.is_public("b/y.py")
    assert not guard.is_public("c/z.py")


def test_a_path_that_cannot_be_redacted_refuses_the_record(tmp_path: Path) -> None:
    """The backstop. Reaching it means the redaction has a bug, and writing the
    file anyway would publish what the guard exists to prevent."""

    class Broken(PublicPathGuard):
        def redact(self, path: str) -> str:
            return path

    with pytest.raises(PrivatePathError):
        Broken([tmp_path / "mine"], [tmp_path]).apply(_record(["theirs/x.cs"]))
