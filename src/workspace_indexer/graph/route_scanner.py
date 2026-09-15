"""What endpoints a file exposes, and what endpoints a file calls.

Rung 1 of #53, and deliberately the same shape as the import scanner's rung 1:
extract the token the source actually wrote, resolve nothing, and report
coverage. Whether resolution is worth building is a question about numbers, and
this is what produces them.

The two halves are not symmetric, and measuring a real ASP.NET + React
workspace before writing anything is what showed how unsymmetric they are.

**Server side is nearly free.** Every `[Route(...)]` carried a literal
template -- 66 of 66 -- and *no* `[HttpGet]`/`[HttpPost]` carried one at all,
so the route always lives on a separate `[Route]` attribute. 17 of those sat on
a class and 36 on a method, and 11 were empty, which in ASP.NET means "exactly
the class route". So an effective route is the class template joined to the
action template, and that join is the whole extraction.

**Client side is where this is won or lost.** Of 32 `fetch` call sites, 3 used
a plain string literal, 11 a template literal, and 18 a variable or expression.
So a scanner that only understands literals sees under a tenth of the calls.
Template literals give up their static prefix, which is why `RouteCall.exact`
exists -- most edges that resolve at all will resolve through a prefix, and
recording them as exact would overstate the graph.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from tree_sitter import Node, Tree

from workspace_indexer.config.graph_section import DEFAULT_HTTP_CLIENTS
from workspace_indexer.graph.parse import parse
from workspace_indexer.graph.route_call import RouteCall
from workspace_indexer.graph.route_declaration import RouteDeclaration
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.graph.routes")

# Languages that can declare an endpoint, and languages that can call one.
# Separate sets because a file is almost never both, and reporting coverage
# for "routes in typescript" would be noise rather than information.
DECLARES = frozenset({"csharp"})
CALLS = frozenset({"typescript", "tsx", "javascript"})
SUPPORTED = DECLARES | CALLS

# Razor's `@page "/path"` directive. Not tree-sitter: a .cshtml file is HTML
# with C# interleaved, and no grammar in the pack parses the combination. The
# directive is the first non-blank line of the file by convention and by
# requirement, which makes a line-anchored regex exact rather than a guess.
_RAZOR_PAGE = re.compile(r'^\s*@page\s+"(?P<template>[^"]*)"', re.MULTILINE)
_PAGE_DIRECTIVE = re.compile(r"^\s*@page\b", re.MULTILINE)

# Method names that count when they hang off a configured object, so `axios`
# in config covers `axios.get(...)` without naming every verb.
_CALL_MEMBERS = frozenset({"get", "post", "put", "delete", "patch", "head", "request"})

# An ASP.NET controller class contributes its name to `[controller]`.
_CONTROLLER_SUFFIX = "Controller"

# Minimal-API registration. `MapGroup` returns a builder carrying a prefix that
# every route registered on it inherits, so it is tracked rather than emitted.
_MAP_VERBS = {
    "MapGet": "GET",
    "MapPost": "POST",
    "MapPut": "PUT",
    "MapDelete": "DELETE",
    "MapPatch": "PATCH",
}
_MAP_GROUP = "MapGroup"

# Receivers that mean "the application root", so a route on one of them has no
# inherited prefix. Names rather than types because the type is not in scope
# here; measured on a real codebase these covered every non-group receiver.
_APP_RECEIVERS = frozenset({"app", "builder", "endpoints", "routes"})


class RouteScanner:
    def __init__(
        self,
        http_clients: Sequence[str] = DEFAULT_HTTP_CLIENTS,
        razor_pages_dir: str = "Pages",
    ) -> None:
        self._clients = frozenset(http_clients)
        self._pages_dir = razor_pages_dir

    def declarations(
        self, text: str, language: str, rel_path: str, tree: Tree | None = None
    ) -> list[RouteDeclaration]:
        """`tree` is the already-parsed source, when the caller has one.

        A C# file is read by three walkers -- imports, namespaces and routes --
        and parsing is the expensive half of each. Passing the tree in is what
        makes that one parse rather than three.

        It must be the tree from parsing *this* `text` as *this* `language`.
        Nothing checks that, and nothing can cheaply: a tree from another
        grammar walks without error and yields no node types this looks for, so
        the failure is an empty result rather than an exception.
        """
        if rel_path.endswith((".cshtml", ".razor")):
            return self._razor(text, rel_path)
        if language not in DECLARES or not text:
            return []
        return self._csharp(text, tree)

    def calls(self, text: str, language: str, tree: Tree | None = None) -> list[RouteCall]:
        if language not in CALLS or not text:
            return []
        if tree is None:
            tree = parse(text, language, log=log)
        if tree is None:
            return []
        found: list[RouteCall] = []
        for node in _walk(tree.root_node):
            if node.type != "call_expression":
                continue
            if not self._is_http_call(node):
                continue
            call = _first_argument_url(node, text)
            if call is not None:
                found.append(call)
        return found

    def _is_http_call(self, node: Node) -> bool:
        """A configured function, or a configured object's request method.

        Names come from config because the useful answer is project-specific:
        one real workspace called `fetch` 27 times and its own `customFetch`
        wrapper 66 times, so a fixed list reported a tenth of the truth as
        complete coverage.
        """
        function = node.child_by_field_name("function")
        if function is None:
            return False
        if function.type == "identifier":
            return function.text is not None and function.text.decode() in self._clients
        if function.type == "member_expression":
            obj = function.child_by_field_name("object")
            prop = function.child_by_field_name("property")
            if obj is None or prop is None or obj.text is None or prop.text is None:
                return False
            if prop.text.decode() not in _CALL_MEMBERS:
                return False
            return obj.text.decode().split(".")[-1] in self._clients
        return False

    # ---- server ---------------------------------------------------------

    def _razor(self, text: str, rel_path: str) -> list[RouteDeclaration]:
        """`@page "template"` when it says one, the file's location otherwise.

        Location is the usual case and was very nearly missed: measured on a
        real workspace, *all twenty* `@page` directives were bare. Razor Pages
        derives the URL from where the file sits, so reading only the directive
        finds nothing and reports it as an absence of routes rather than an
        absence of templates.
        """
        explicit = [
            RouteDeclaration(
                template=match.group("template"),
                method=None,
                line=text[: match.start()].count("\n") + 1,
                kind="page",
            )
            for match in _RAZOR_PAGE.finditer(text)
            if match.group("template").strip()
        ]
        if explicit:
            return explicit
        if not _PAGE_DIRECTIVE.search(text):
            # A .cshtml with no @page at all is a layout, a partial or a view
            # component. It is not addressable, so it declares nothing.
            return []
        template = self._route_from_path(rel_path)
        if template is None:
            return []
        line = 1
        found = _PAGE_DIRECTIVE.search(text)
        if found is not None:
            line = text[: found.start()].count("\n") + 1
        return [RouteDeclaration(template=template, method=None, line=line, kind="page")]

    def _route_from_path(self, rel_path: str) -> str | None:
        """`.../Pages/Remittance/Detail.cshtml` -> `Remittance/Detail`.

        `Index` is dropped, matching the convention that `Pages/Foo/Index`
        answers `/Foo`. Anything outside the pages directory has no
        conventional route and returns None rather than a guess.
        """
        parts = rel_path.replace("\\", "/").split("/")
        if self._pages_dir not in parts:
            return None
        tail = parts[parts.index(self._pages_dir) + 1 :]
        if not tail:
            return None
        stem = tail[-1].rsplit(".", 1)[0]
        segments = [*tail[:-1], *([] if stem.lower() == "index" else [stem])]
        return "/".join(segments)

    def _csharp(self, text: str, tree: Tree | None = None) -> list[RouteDeclaration]:
        if tree is None:
            tree = parse(text, "csharp", log=log)
        if tree is None:
            return []
        return self._controllers(tree.root_node, text) + self._minimal_apis(tree.root_node, text)

    def _minimal_apis(self, root: Node, text: str) -> list[RouteDeclaration]:
        """`app.MapGet("/x", ...)`, including routes registered on a group.

        A second ASP.NET style, and on the codebase this was measured against
        the *only* one: 1037 C# files, zero `[Route]` attributes, 140
        `Map*` calls. The two styles turned out to be mutually exclusive per
        project, so supporting one is supporting half the stack.

        `MapGroup` returns a builder carrying a prefix, and every route
        registered on it inherits that prefix. Emitting the leaf alone would
        give `{id}` as a route -- not merely incomplete but wrong, since it
        would match paths that belong to something else. 117 of 124 leaf calls
        measured were registered on such a group, so this is the common case
        rather than an edge one.
        """
        groups = self._group_prefixes(root, text)
        found: list[RouteDeclaration] = []
        for node in _walk(root):
            if node.type != "invocation_expression":
                continue
            name, receiver = _member_call(node, text)
            verb = _MAP_VERBS.get(name or "")
            if verb is None or receiver is None:
                continue
            template = _string_argument(node, text)
            if template is None:
                # The path came from a variable or an expression. One of 140
                # measured; recording a guess would be worse than the gap.
                continue
            prefix = _prefix_of(receiver, text, groups)
            if prefix is None:
                continue
            joined = _join(prefix, template)
            if not joined:
                continue
            found.append(
                RouteDeclaration(
                    template=joined,
                    method=verb,
                    line=node.start_point[0] + 1,
                    kind="minimal",
                )
            )
        return found

    def _group_prefixes(self, root: Node, text: str) -> dict[str, str]:
        """Variable name -> the prefix routes registered on it inherit.

        Within one file only. That is not a simplification standing in for
        something better: a group is a local, and C# requires it declared
        before use, so a single ordered pass resolves nesting
        (`var sub = parent.MapGroup("more")`) without any dataflow analysis.
        """
        groups: dict[str, str] = {}
        # Source order, explicitly. `_walk` returns nodes in traversal order,
        # which is not textual order, so a nested group could be visited before
        # the group it nests inside and resolve to nothing. C# requires a local
        # declared before use, so sorting by position is all the ordering this
        # needs -- and it is the whole reason no dataflow analysis is required.
        declarators = sorted(
            (n for n in _walk(root) if n.type == "variable_declarator"),
            key=lambda n: n.start_byte,
        )
        for node in declarators:
            name = _identifier(node, text)
            initialiser = _initialiser(node)
            if not name or initialiser is None:
                continue
            prefix = _group_prefix_of(initialiser, text, groups)
            if prefix is not None:
                groups[name] = prefix
        return groups

    def _controllers(self, root: Node, text: str) -> list[RouteDeclaration]:
        found: list[RouteDeclaration] = []
        for klass in _walk(root):
            if klass.type != "class_declaration":
                continue
            name = _identifier(klass, text)
            prefix = _attribute_string(klass, "Route", text, nested=False)
            if prefix is None:
                # No class-level route. Actions can still carry a whole route
                # of their own, so this is not a reason to stop.
                prefix = ""
            prefix = _expand_controller(prefix, name)

            for method in _walk(klass):
                if method.type != "method_declaration":
                    continue
                action = _attribute_string(method, "Route", text, nested=False)
                verb = _http_verb(method, text)
                if action is None and verb is None:
                    continue
                template = _join(prefix, action or "")
                if not template:
                    continue
                found.append(
                    RouteDeclaration(
                        template=template,
                        method=verb,
                        line=method.start_point[0] + 1,
                        kind="controller",
                    )
                )
        return found


# ---- helpers ------------------------------------------------------------


def _walk(node: Node) -> list[Node]:
    out: list[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(current.children)
    return out


def _text(node: Node, source: str) -> str:
    return source.encode()[node.start_byte : node.end_byte].decode(errors="replace")


def _identifier(node: Node, source: str) -> str:
    for child in node.children:
        if child.type == "identifier":
            return _text(child, source)
    return ""


def _attribute_string(owner: Node, name: str, source: str, *, nested: bool) -> str | None:
    """The string argument of `[name("...")]` declared directly on `owner`.

    Direct children only. A class's attribute list and its methods' attribute
    lists are both inside the class node, so walking the whole subtree would
    hand every method's route to the class and produce one enormous wrong
    prefix for all of them.
    """
    for child in owner.children:
        if child.type != "attribute_list":
            continue
        for attribute in child.children:
            if attribute.type != "attribute":
                continue
            if _identifier(attribute, source) != name:
                continue
            for part in _walk(attribute):
                if part.type == "string_literal":
                    return _text(part, source).strip('"')
    _ = nested
    return None


def _http_verb(method: Node, source: str) -> str | None:
    for child in method.children:
        if child.type != "attribute_list":
            continue
        for attribute in child.children:
            if attribute.type != "attribute":
                continue
            name = _identifier(attribute, source)
            if name.startswith("Http") and len(name) > 4:
                return name[4:].upper()
    return None


def _expand_controller(template: str, class_name: str) -> str:
    """`[controller]` is the class name with "Controller" removed.

    Expanded here rather than at match time because it is a fact about the
    declaration, not about the comparison -- and leaving it in would make every
    controller's route look identical to every other's.
    """
    if "[controller]" not in template:
        return template
    stem = class_name.removesuffix(_CONTROLLER_SUFFIX) or class_name
    return template.replace("[controller]", stem)


def _join(prefix: str, action: str) -> str:
    parts = [part.strip("/") for part in (prefix, action) if part.strip("/")]
    return "/".join(parts)


def _first_argument_url(call: Node, source: str) -> RouteCall | None:
    """As much of the first argument as is knowable without running the code.

    A plain string is the whole URL. A template literal gives up the text
    before its first interpolation, which is enough to name an endpoint but not
    to tell two routes sharing a prefix apart -- hence `exact=False`. Anything
    else is a variable, and inventing a URL for it would be worse than
    recording nothing.
    """
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return None
    for argument in arguments.children:
        if argument.type == "string":
            for part in argument.children:
                if part.type == "string_fragment":
                    return RouteCall(
                        target=_text(part, source),
                        line=call.start_point[0] + 1,
                        exact=True,
                    )
            return None
        if argument.type == "template_string":
            first = argument.children[1] if len(argument.children) > 1 else None
            if first is None or first.type != "string_fragment":
                # Interpolation first (`${BASE}/api/x`): the static part names
                # no endpoint on its own.
                return None
            return RouteCall(
                target=_text(first, source),
                line=call.start_point[0] + 1,
                exact=False,
            )
        if argument.type in ("(", ")", ","):
            continue
        # A variable or an expression. 18 of 32 call sites in the workspace
        # measured, and the reason rung 3 exists.
        return None
    return None


def _initialiser(declarator: Node) -> Node | None:
    """The expression a `var x = ...` binds.

    Fetched by position rather than by field name: tree-sitter's C# grammar
    leaves it an unnamed child, so `child_by_field_name("value")` returns None
    for every declarator. Asking for the field and believing the answer said
    "no minimal-API routes use a group variable" when 117 of 124 do.
    """
    for child in declarator.children:
        if child.type not in ("identifier", "="):
            return child
    return None


def _member_call(node: Node, source: str) -> tuple[str | None, Node | None]:
    """`(method name, receiver)` for `receiver.Method(...)`."""
    function = node.child_by_field_name("function")
    if function is None or function.type != "member_access_expression":
        return None, None
    name = function.child_by_field_name("name")
    receiver = function.child_by_field_name("expression")
    if name is None:
        return None, receiver
    return _text(name, source), receiver


def _string_argument(node: Node, source: str) -> str | None:
    """The first argument, when it is a string literal."""
    arguments = node.child_by_field_name("arguments")
    if arguments is None:
        return None
    for argument in arguments.children:
        if argument.type in ("(", ")", ","):
            continue
        for part in _walk(argument):
            if part.type == "string_literal":
                return _text(part, source).strip('"')
        return None
    return None


def _group_prefix_of(node: Node, source: str, groups: dict[str, str]) -> str | None:
    """The prefix a `MapGroup(...)` expression establishes, or None.

    Looks *through* the builder calls that idiomatically follow it --
    `app.MapGroup("/x").WithTags("X").RequireAuthorization()` is one
    expression whose outermost invocation is `RequireAuthorization`, not
    `MapGroup`. Insisting on the outermost call being MapGroup found 19
    endpoints where the codebase has around 140, and the gap was entirely
    this: groups configured on the same line they are created.
    """
    current = node
    while current.type == "invocation_expression":
        name, receiver = _member_call(current, source)
        if receiver is None:
            return None
        if name == _MAP_GROUP:
            literal = _string_argument(current, source)
            if literal is None:
                return None
            parent = _prefix_of(receiver, source, groups)
            return None if parent is None else _join(parent, literal)
        if name in _MAP_VERBS:
            # A route, not a group. Whatever this expression is, it does not
            # establish a prefix for anything else.
            return None
        current = receiver
    return None


def _prefix_of(receiver: Node, source: str, groups: dict[str, str]) -> str | None:
    """What a route registered on this receiver inherits.

    Empty string for the application root, the group's prefix for a group, and
    None for anything this cannot follow -- a field, a parameter, a builder
    from another file. None skips the route rather than emitting it with a
    missing prefix, because a route missing its prefix is not incomplete, it
    is wrong: it would match paths belonging to something else.
    """
    if receiver.type == "identifier":
        name = _text(receiver, source)
        if name in groups:
            return groups[name]
        return "" if name in _APP_RECEIVERS else None
    if receiver.type == "invocation_expression":
        # Chained: `app.MapGroup("/a").MapGet(...)`.
        return _group_prefix_of(receiver, source, groups)
    return None
