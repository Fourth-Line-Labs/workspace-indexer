"""Whether an import names the language's own standard library.

Kept apart from declared dependencies because the evidence differs in kind: a
package id is read from a manifest the build already depends on, while this is
a membership test against a list that ships with the language. Where the two
overlap -- `System.Text.Json` is both a BCL namespace and a NuGet package --
the declared one wins, because declared data beats a prefix.

Everything unrecognised returns False so it lands in `UNCLASSIFIED`, where the
gap is visible and countable. A permissive default would put it in `FRAMEWORK`,
which is the same edge reported as correctly-unresolvable when nobody checked.
"""

from __future__ import annotations

import sys

# Node's own modules. There is no runtime list to consult from Python, so this
# is maintained by hand and deliberately conservative at the edges: a missing
# entry becomes UNCLASSIFIED, which is visible, rather than FRAMEWORK, which
# would be silently wrong.
_NODE_BUILTINS = frozenset(
    {
        "assert", "async_hooks", "buffer", "child_process", "cluster",
        "console", "constants", "crypto", "dgram", "diagnostics_channel",
        "dns", "domain", "events", "fs", "http", "http2", "https",
        "inspector", "module", "net", "os", "path", "perf_hooks", "process",
        "punycode", "querystring", "readline", "repl", "stream",
        "string_decoder", "sys", "timers", "tls", "trace_events", "tty",
        "url", "util", "v8", "vm", "wasi", "worker_threads", "zlib",
    }
)  # fmt: skip

_JS_LANGUAGES = frozenset({"javascript", "typescript", "tsx"})


def is_framework_module(module: str, language: str) -> bool:
    """True when `module` names something that ships with `language`.

    Note the Python answer is taken from *this* interpreter's standard library,
    not the indexed project's. The list barely moves between versions and the
    alternative is shipping a per-version table, so the inaccuracy is accepted
    and recorded rather than engineered around.
    """
    if not module:
        return False
    if language == "python":
        # `os.path` is stdlib because `os` is; the list holds top-level names.
        # `test` is in it, so a project with its own top-level `test` package
        # would land here -- which is why resolved and relative edges are
        # claimed as first-party before this is ever consulted.
        return module.split(".", 1)[0] in sys.stdlib_module_names
    if language == "csharp":
        # The BCL is one namespace root. `Microsoft.*` is deliberately not
        # treated as framework: some of it is the BCL, some ships as packages,
        # and much arrives via the shared framework with no PackageReference
        # anywhere. Left for the declared-dependency rule to claim, and
        # otherwise counted as unclassified rather than guessed at.
        return module == "System" or module.startswith("System.")
    if language in _JS_LANGUAGES:
        # `node:fs` is the explicit spelling of `fs`; `fs/promises` is a
        # subpath of a builtin.
        bare = module.removeprefix("node:")
        return bare.split("/", 1)[0] in _NODE_BUILTINS
    return False
