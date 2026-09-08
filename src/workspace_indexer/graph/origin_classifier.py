"""Which bucket an import edge belongs to.

Order is precedence, and it is the whole design:

1. **Resolved** edges are first-party by construction -- they already point at
   an indexed file.
2. A **relative** specifier names a neighbour, so it is first-party whether or
   not resolution managed it. This is the bucket that makes resolver bugs
   visible: an unresolved relative import is always ours to fix.
3. A **declared dependency**, read from a manifest the build already depends
   on. Ahead of the framework test because declared data beats a prefix, and
   because the two genuinely overlap -- a project that declares
   `System.Text.Json` really does get it from a package.
4. The language's **standard library**.
5. Nothing claimed it, which is reported rather than absorbed.

Declared dependencies are scoped per `(root_label, unit)`, the same key the
resolver uses, for the same reason: two repositories in one workspace have
different dependency sets, and one repo's `package.json` says nothing about
another's imports.
"""

from __future__ import annotations

from collections.abc import Mapping

from workspace_indexer.graph.framework_modules import JS_LANGUAGES, is_framework_module
from workspace_indexer.graph.import_origin import ImportOrigin
from workspace_indexer.graph.unit import unit_of


class OriginClassifier:
    def __init__(
        self, dependencies: Mapping[tuple[str, str], frozenset[str]] | None = None
    ) -> None:
        """`dependencies` maps (root_label, unit) to the package ids that unit
        declares. An absent or empty entry is not an error -- it means nothing
        has been read for that unit yet, and its third-party imports will
        report as unclassified until something does.
        """
        self._dependencies: dict[tuple[str, str], frozenset[str]] = dict(dependencies or {})

    def classify(
        self,
        module: str,
        *,
        language: str,
        is_relative: bool,
        resolved: str | None,
        root_label: str,
        from_path: str,
    ) -> ImportOrigin:
        if resolved is not None or is_relative:
            return ImportOrigin.FIRST_PARTY
        if self._is_declared(module, root_label, unit_of(from_path), language):
            return ImportOrigin.DECLARED_DEPENDENCY
        if is_framework_module(module, language):
            return ImportOrigin.FRAMEWORK
        return ImportOrigin.UNCLASSIFIED

    def _is_declared(self, module: str, root_label: str, unit: str, language: str) -> bool:
        """Exact id, or the id followed by a separator that means "inside it".

        The separator is required rather than a bare prefix, or `lodash` would
        claim `lodashy`. Which separators count is language-dependent, because
        a dot does not mean the same thing in every registry:

        - `/` everywhere. `lodash/debounce` and `@scope/pkg/sub` are subpaths.
        - `.` only outside the JS family. `MyPackage.Sub` is inside
          `MyPackage` in .NET and Python, but on npm `lodash.merge` is an
          independently published package, so claiming it for a unit that
          declared `lodash` would hide an undeclared dependency from the
          unclassified queue -- the one bucket that has to stay honest.
        """
        declared = self._dependencies.get((root_label, unit))
        if not declared:
            return False
        separators = ("/",) if language in JS_LANGUAGES else (".", "/")
        return any(
            module == package or any(module.startswith(f"{package}{sep}") for sep in separators)
            for package in declared
        )
