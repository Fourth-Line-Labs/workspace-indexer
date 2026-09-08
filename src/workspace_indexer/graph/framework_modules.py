"""Whether an import names the language's own standard library.

Kept apart from declared dependencies because the evidence differs in kind: a
package id is read from a manifest the build already depends on, while this is
a membership test against a list that ships with the language. Where the two
overlap the declared one wins, because declared data beats a prefix -- and they
do overlap: `System.Text.Json` is part of .NET's standard library *and* a
NuGet package you can reference explicitly.

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

# Builtins that exist only as `node:x`. `module.builtinModules` lists
# `node:test` and not `test`, and `require("test")` resolves to the npm
# package of that name -- so matching these bare would claim a real
# dependency as framework, which is the false positive this module's
# conservatism exists to avoid.
_NODE_PREFIX_ONLY = frozenset({"test", "sqlite", "sea"})

# Public because the origin classifier needs the same set, and a second copy
# would drift the moment one of them learns about a new dialect.
JS_LANGUAGES = frozenset({"javascript", "typescript", "tsx"})


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
        # .NET's standard library sits under one namespace root, so this is a
        # prefix test rather than a list.
        #
        # `Microsoft.*` is deliberately NOT treated as framework, because the
        # prefix tells you nothing about where the code comes from. It spans
        # three different origins at once: `Microsoft.Win32.*` is standard
        # library, `Microsoft.Extensions.*` is mostly NuGet packages, and
        # `Microsoft.AspNetCore.*` arrives through the shared framework that
        # the Web SDK references implicitly -- so it appears in no
        # PackageReference anywhere. One prefix, three answers. Left for the
        # declared-dependency rule to claim where a manifest names it, and
        # otherwise counted as unclassified rather than guessed at.
        return module == "System" or module.startswith("System.")
    if language in JS_LANGUAGES:
        # `node:fs` is the explicit spelling of `fs`; `fs/promises` is a
        # subpath of a builtin.
        prefixed = module.startswith("node:")
        head = module.removeprefix("node:").split("/", 1)[0]
        if head in _NODE_BUILTINS:
            return True
        # Prefix-only builtins are framework *only* when written that way.
        return prefixed and head in _NODE_PREFIX_ONLY
    return False
