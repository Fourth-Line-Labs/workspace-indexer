"""Measuring the language fixture trees, and saying what moved when they change.

Everything here goes through the same public manifest API an agent reaches
through -- `imports_of`, `namespaces_of`, `dependencies_of` -- rather than
through SQL. Two reasons: a gate that queries the database directly can pass
while the API that everything actually uses is broken, and grouping by the
fixture's own directory rather than by the manifest's `language` column means a
file that lands in the wrong language bucket shows up as a difference instead
of being counted under whatever the manifest decided.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.language_baseline import LanguageBaseline
from workspace_indexer.state import Manifest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "languages"
BASELINES = FIXTURE_ROOT / "baselines.json"


def fixture_languages() -> list[str]:
    """The language directories that exist, which is the fixture set itself."""
    return sorted(p.name for p in FIXTURE_ROOT.iterdir() if p.is_dir())


def source_files() -> list[str]:
    """Every fixture source file, as a path relative to the fixture root.

    READMEs and the baselines file are excluded: they document the corpus
    rather than belonging to it, and indexing them would put their prose into
    the counts.
    """
    return sorted(
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file() and path.name not in ("README.md", "baselines.json")
    )


def measure(
    manifest: Manifest, *, root_label: str
) -> tuple[dict[str, LanguageBaseline], list[str]]:
    """The counts per language, and the files the run did not index.

    The second half is the one with teeth. A file withheld by the secret
    scanner leaves no row anywhere -- that is the point of withholding -- so
    the only way to notice is to compare what is on disk against what was
    recorded. `CLAUDE.md` calls a false positive there silent data loss, and
    this is what makes it audible.
    """
    indexed = {
        rel_path
        for _, rel_path, _ in manifest.indexed_documents()
        if rel_path in set(source_files())
    }
    withheld = [rel for rel in source_files() if rel not in indexed]

    measured: dict[str, LanguageBaseline] = {}
    for language in fixture_languages():
        files = sorted(rel for rel in indexed if rel.split("/")[0] == language)
        edges = 0
        with_imports = 0
        declarations = 0
        buckets = {
            "first_party": 0,
            "declared_dependency": 0,
            "framework": 0,
            "unclassified": 0,
        }
        resolved_first_party = 0
        declined = 0

        for rel in files:
            statements = manifest.imports_of(root_label, rel)
            edges += len(statements)
            with_imports += 1 if statements else 0
            declarations += len(manifest.namespaces_of(root_label, rel))

            # One entry per *edge*, which is what the origin columns count --
            # a namespace edge expands to one row per declaring file in
            # `dependencies_of`, so the expansion is collapsed on the module.
            seen: dict[tuple[str, int], tuple[str | None, bool, str | None]] = {}
            for dependency in manifest.dependencies_of(root_label, rel):
                key = (dependency.module, dependency.line)
                was = seen.get(key)
                resolved = dependency.resolved or (was[1] if was else False)
                seen[key] = (dependency.origin, resolved, dependency.resolved_by)
            for origin, resolved, resolved_by in seen.values():
                if resolved_by == "declined":
                    declined += 1
                    continue
                if origin in buckets:
                    buckets[origin] += 1
                if origin == "first_party" and resolved:
                    resolved_first_party += 1

        measured[language] = LanguageBaseline(
            files_indexed=len(files),
            files_with_imports=with_imports,
            import_edges=edges,
            namespace_declarations=declarations,
            first_party=buckets["first_party"],
            first_party_resolved=resolved_first_party,
            declared_dependency=buckets["declared_dependency"],
            framework=buckets["framework"],
            unclassified=buckets["unclassified"],
            declined=declined,
        )
    return measured, withheld


def load_baselines() -> tuple[dict[str, LanguageBaseline], list[str]]:
    recorded = json.loads(BASELINES.read_text(encoding="utf-8"))
    languages = {
        name: LanguageBaseline.model_validate(row) for name, row in recorded["languages"].items()
    }
    return languages, list(recorded["withheld"])


def describe(measured: dict[str, LanguageBaseline], expected: dict[str, LanguageBaseline]) -> str:
    """The failure message, written for someone who has to act on it.

    "expected 1460, got 1455" sends a reader spelunking. Naming the language,
    the field, the direction and the fixture path is the difference between a
    gate people keep and a gate people disable.
    """
    lines: list[str] = []
    for language in sorted(set(measured) | set(expected)):
        if language not in expected:
            lines.append(f"{language}: no baseline row (fixtures/languages/{language}/ is new)")
            continue
        if language not in measured:
            lines.append(f"{language}: baseline row with no fixture directory")
            continue
        was, now = expected[language], measured[language]
        for field in LanguageBaseline.model_fields:
            before, after = getattr(was, field), getattr(now, field)
            if before != after:
                lines.append(
                    f"{language}.{field}: {before} -> {after} "
                    f"({after - before:+d}), see tests/fixtures/languages/{language}/"
                )
    return "\n".join(lines)
