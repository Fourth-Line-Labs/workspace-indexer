"""Turning an import string into a file that exists.

Rung 2 of #34. Rung 1 recorded what the source wrote; this decides which
indexed file it meant, for the cases that can be settled without a build
system.

Resolution is scoped to the importing file's own unit -- its repository. An
import inside one repo names something in that repo, and reaching across
repositories needs a workspace-wide symbol table, which is rung 3. Keeping the
scope narrow is also what makes an answer trustworthy: two repos routinely
contain the same package name, and a resolver that guesses between them is
worse than one that declines.

What it deliberately does not do:

- **tsconfig path aliases.** `@/hooks/useThing` needs tsconfig.json parsed and
  its `paths` applied relative to `baseUrl`. Real work, and until it exists an
  alias resolves to nothing rather than to something plausible.
- **node_modules.** `react` is a package, not a file in this workspace.
- **C# namespaces.** `using MyApp.Data` names no path at all; it needs every
  `namespace` declaration collected first.
"""

from __future__ import annotations

from workspace_indexer.graph.framework_modules import JS_LANGUAGES
from workspace_indexer.graph.import_edge import ImportEdge
from workspace_indexer.graph.unit import unit_of
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.graph.resolve")

# Tried in order for a JS/TS specifier with no extension. `.tsx` before `.ts`
# would resolve a component to the wrong file where both exist, which is rare
# but silent, so the order follows what a bundler does.
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_JS_INDEXES = tuple(f"/index{ext}" for ext in _JS_EXTENSIONS)

# TypeScript's ESM output convention: source written as `import './x.js'`
# compiles from `x.ts`. The specifier names the emitted file, which is not the
# file in the repository, so a literal lookup finds nothing.
_ESM_REWRITES = {
    ".js": (".ts", ".tsx"),
    ".mjs": (".mts",),
    ".cjs": (".cts",),
}

# Shared with the origin classifier rather than redefined. The two have to
# agree: this decides an edge is JS and resolves it as a path, while the
# classifier decides `.` is not a subpath separator for the same edge. Two
# copies means adding a dialect to one makes them disagree, and both keep
# returning confident answers while they do.


class ImportResolver:
    """Resolves edges against the set of files actually indexed.

    That set is the whole point: it spans every repository in the workspace,
    which is what a per-project language server does not have.
    """

    def __init__(self, files_by_unit: dict[tuple[str, str], set[str]]) -> None:
        # rel_path is unique per unit, so a set membership test is the whole
        # lookup. Suffix matching needs the paths themselves, kept per unit so
        # one repo cannot resolve into another.
        self._files = files_by_unit

    def resolve(
        self, edge: ImportEdge, *, from_path: str, root_label: str, language: str
    ) -> str | None:
        unit = unit_of(from_path)
        known = self._files.get((root_label, unit))
        if not known:
            return None
        if language == "python":
            return self._python(edge, from_path, known)
        if language in JS_LANGUAGES:
            return self._javascript(edge, from_path, known)
        return None

    # ---- python ---------------------------------------------------------

    def _python(self, edge: ImportEdge, from_path: str, known: set[str]) -> str | None:
        if edge.is_relative:
            return self._python_relative(edge.module, from_path, known)
        # `workspace_indexer.models` -> a path ending `workspace_indexer/models`.
        # Matching by suffix rather than by sys.path means no interpreter
        # configuration has to be reproduced, and it is right whenever the
        # package directory is named after the package -- which is the
        # convention this resolves for.
        return _first_suffix_match(edge.module.replace(".", "/"), known)

    def _python_relative(self, module: str, from_path: str, known: set[str]) -> str | None:
        """`.helpers` and `..db.models`, resolved against the importing file.

        Leading dots count levels up: one dot is the containing package, two is
        its parent, and so on.
        """
        dots = len(module) - len(module.lstrip("."))
        remainder = module[dots:]
        parts = from_path.split("/")[:-1]  # the importing file's directory
        # One dot means "this package", so only the extra dots walk upward.
        if dots > 1:
            parts = parts[: -(dots - 1)] if dots - 1 <= len(parts) else []
        base = "/".join([*parts, *(remainder.split(".") if remainder else [])])
        return _python_candidates(base, known)

    # ---- javascript family ----------------------------------------------

    def _javascript(self, edge: ImportEdge, from_path: str, known: set[str]) -> str | None:
        if not edge.is_relative:
            # A bare specifier is a package or a tsconfig alias. Both need
            # something we do not have yet, and guessing would be worse.
            return None
        parts = from_path.split("/")[:-1]
        for segment in edge.module.split("/"):
            if segment == ".":
                continue
            if segment == "..":
                parts = parts[:-1]
            else:
                parts.append(segment)
        base = "/".join(parts)
        if base in known:
            return base

        # `./x.js` may name `x.ts`. Tried before the plain extension sweep,
        # because appending to a specifier that already has an extension would
        # look for `x.js.ts`.
        for suffix, replacements in _ESM_REWRITES.items():
            if base.endswith(suffix):
                stem = base[: -len(suffix)]
                for candidate in (f"{stem}{ext}" for ext in replacements):
                    if candidate in known:
                        return candidate

        for candidate in (f"{base}{ext}" for ext in _JS_EXTENSIONS):
            if candidate in known:
                return candidate
        for candidate in (f"{base}{index}" for index in _JS_INDEXES):
            if candidate in known:
                return candidate
        return None


def _python_candidates(base: str, known: set[str]) -> str | None:
    for candidate in (f"{base}.py", f"{base}/__init__.py", f"{base}.pyi"):
        if candidate in known:
            return candidate
    return None


def _first_suffix_match(module_path: str, known: set[str]) -> str | None:
    """The one indexed file whose path ends with this module path.

    Exactly one, deliberately. Two matches means the module name is ambiguous
    within the repository, and picking one would produce a confident edge
    pointing at the wrong file -- worse than no edge, because nothing
    downstream could tell.
    """
    suffixes = (f"/{module_path}.py", f"/{module_path}/__init__.py")
    matches = [path for path in known if path.endswith(suffixes)]
    if len(matches) == 1:
        return matches[0]
    return None
