"""Refuse to embed files that contain credentials.

Indexed content is sent to an embedding provider as request input, so anything
reaching a chunk reaches a third party. `.gitignore` covers the conventional
cases and misses three real ones: non-repo folders have no ignore file at all,
committed secrets are tracked by definition, and the case that actually bit
this project was a file named `.mcp.json` holding a GitHub token — a name no
deny-list would flag.

So the check is on content, not on filenames, and it protects source files as
much as configuration.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from workspace_indexer.secrets.secret_finding import SecretFinding

# Token shapes that are unambiguous: these prefixes are issued by a provider
# and do not occur by accident.
_SIGNATURES: list[tuple[str, re.Pattern[str], str]] = [
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "GitHub personal access token"),
    ("github_classic", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "OpenAI-style API key"),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "AWS access key id"),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API key"),
    ("voyage", re.compile(r"\bpa-[A-Za-z0-9_-]{30,}"), "Voyage API key"),
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"), "private key block"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."), "JWT"),
]

# An assignment to a credential-shaped name. The value still has to look random
# before we act on it, or every `API_KEY=` line in a template trips the check.
#
# Two shapes this missed, both found against a real repository:
#
#   "password": "..."     A quoted key. The pattern ran key straight into
#                         `[:=]`, so JSON -- the single commonest way a
#                         credential is written down -- never matched at all.
#   PASSWORD="a~b"        `~` was absent from the value class. Azure service
#                         principal passwords routinely contain it, and the
#                         value simply failed to match rather than failing the
#                         entropy test.
#
# The value class is now "printable, not whitespace or a quote", which is what
# a generated credential actually looks like. Entropy remains the thing that
# decides, so widening this cannot by itself produce a false positive.
#
# `??` counts as an assignment operator because a C# null-coalescing fallback
# *is* one -- `ApiKey = options?.ApiKey ?? "literal"` gives `ApiKey` that
# literal whenever the left side is null. Before the `?.` fix above, a line of
# this shape was flagged for the wrong reason: the scanner read `options?.ApiKey`
# as a generated value and withheld the file, which happened to cover the real
# literal after the `??`. Teaching it to see the expression would otherwise have
# quietly uncovered that case.
#
# `&` ends a value because a query string is a run of assignments, not one:
# `?authSource=admin&directConnection=true&...` was reading everything after
# `authSource=` as that parameter's value, which cleared the entropy bar and
# withheld a file over a parameter naming a database. The credential in that
# same URL sits in the authority and is caught by `_URL_CREDENTIAL` below.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.-]*
        (?:api[_-]?key|secret|token|password|passwd|credential|auth|
           connection[_-]?string|conn[_-]?str|sas|pwd)
        [A-Za-z0-9_.-]*)
    ["']?              # a quoted key: "password": "..."
    \s* (?: [:=] | \?\? ) \s*   # `??` assigns a default in C#, the same way
    ["']?
    (?P<value>[^\s"'`,;&)\]}]{16,})
    """
)

# Word separators inside an identifier. `docker-hub-credentials` is the *name*
# of a credential, not the credential; stripping these before the all-letters
# test recognises it as a name rather than a generated value.
_WORD_SEPARATORS = str.maketrans("", "", "-_")


# A credential inline in a URL's authority: `scheme://user:secret@host`.
#
# No assignment rule can see this -- there is no `key = value` -- so a
# connection string with the password in it was passing through untouched. The
# one that prompted this was caught only because an unrelated query parameter
# tripped the entropy rule; with the query string removed it would have been
# embedded and sent to the provider.
#
# Entropy is deliberately *not* consulted here. A weak password is still a
# password, and unlike a bare assignment the shape itself is the evidence:
# nobody writes `user:value@host` for anything but a credential.
_URL_CREDENTIAL = re.compile(
    r"(?i)\b[a-z][a-z0-9+.\-]*://(?P<user>[^\s:/?#@\[\]]+):(?P<password>[^\s:/?#@\[\]]+)@"
)

# Passwords that are obviously the *shape* of a credential rather than one.
# Documentation showing connection-string syntax is exactly the content this
# index exists to retrieve, so withholding a page over `user:password@host`
# would be the destructive error. Compared after stripping separators and
# lowercasing, so `Your_Password` and `your-password` are both covered.
_URL_PLACEHOLDER_PASSWORDS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "pass",
        "secret",
        "token",
        "apikey",
        "key",
        "credential",
        "credentials",
        "user",
        "username",
        "admin",
        "root",
        "mypassword",
        "yourpassword",
        "example",
        "redacted",
        "hidden",
    }
)


# Values below this Shannon entropy read as prose or a placeholder rather than
# a generated credential.
_ENTROPY_THRESHOLD = 3.6

# Obvious non-secrets that would otherwise clear the entropy bar.
_PLACEHOLDERS = frozenset(
    {
        "changeme",
        "your-api-key-here",
        "xxxxxxxxxxxxxxxx",
        "insert-key-here",
        "replace-me",
        "todo",
        "none",
        "null",
        "example",
        "placeholder",
    }
)


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


# Code referring to a credential rather than the credential itself, as in
# `api_key=settings.qdrant_api_key`. Found by this scanner withholding one of
# our own source files.
#
# Identifier *validity* is the wrong test -- a generated secret like
# k7Fq2mZx9RtVwLpA3nBcYdQe is a valid identifier too. What separates them is
# naming convention: references are dotted, or consistently snake_case or
# SCREAMING_CASE. A generated value interleaves case and digits with no word
# structure at all.
_ATTRIBUTE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z")
_CONVENTIONAL_NAME = re.compile(r"\A(?:[a-z_][a-z0-9_]*|[A-Z_][A-Z0-9_]*|[a-zA-Z][a-zA-Z]*)\Z")

# An expression or a type rather than a literal. Widening the value class to
# catch `~` started catching all of these, and withholding a whole file over a
# variable reference is a silent loss of content -- the worse error here.
#
#   builder.Configuration["x"]          brackets
#   AuthMode::TrustedLocal              a Rust or C++ path
#   Option<AuthMeta>                    a generic type
#   appInsights.properties.Connection   a dotted path
#   options?.MaxOutputTokens            a null-conditional member access
#   x ?? _fallback                      a null-coalescing default
#
# The last two are C#, and the anchored alternative could not see them: it
# wants an identifier followed immediately by a dot, and `?.` puts a `?`
# there. An ordinary object initializer -- `Tokens = options?.MaxTokens ?? _t,`
# -- therefore read as a generated credential, because the key matched on
# `token` and nothing rejected the value. Eight tracked files were purged from
# a live index over this shape.
#
# `?.` and `??` are matched anywhere in the value rather than only at the
# start, because unlike a bare dot they cannot occur inside a credential: a
# generated value is drawn from an alphabet, and no provider issues one
# containing `?`. The dotted alternative stays anchored for the opposite
# reason -- a dot *can* appear in a credential, so accepting one anywhere
# would be a way to hide a real value.
_EXPRESSION = re.compile(r"[()\[\]{}<>]|::|\?[?.]|\A[A-Za-z_][A-Za-z0-9_]*\.")


def _is_placeholder_password(password: str) -> bool:
    """A sample, not a secret: `<password>`, `${PASSWORD}`, `%PW%`, `****`."""
    if password.startswith(("<", "${", "%", "{")):
        return True
    bare = password.translate(_WORD_SEPARATORS).lower()
    if bare in _URL_PLACEHOLDER_PASSWORDS or bare in _PLACEHOLDERS:
        return True
    # A run of one masking character, as a page redacting its own example.
    return len(set(password)) == 1 and not password.isalnum()


def _looks_generated(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDERS or lowered.startswith("<") or lowered.startswith("${"):
        return False
    # A path, a URL or a dotted module name is structured, not random.
    if "/" in value or value.count(".") > 2:
        return False
    bare = value.strip()
    if _ATTRIBUTE.match(bare) or _CONVENTIONAL_NAME.match(bare):
        return False
    if _EXPRESSION.search(bare):
        return False
    # A generated credential mixes letters with digits or symbols. All-letters
    # is an identifier -- `sqlAdminPassword`, `docker-hub-credentials` --
    # however long and however cased, and both clear the entropy bar easily.
    if bare.translate(_WORD_SEPARATORS).isalpha():
        return False
    return shannon_entropy(value) >= _ENTROPY_THRESHOLD


def _signature(line: str, number: int) -> SecretFinding | None:
    for rule, pattern, description in _SIGNATURES:
        if pattern.search(line):
            return SecretFinding(rule=rule, line=number, description=description)
    return None


def _url_credential(line: str, number: int) -> SecretFinding | None:
    match = _URL_CREDENTIAL.search(line)
    if match is None or _is_placeholder_password(match.group("password")):
        return None
    return SecretFinding(
        rule="url_credential",
        line=number,
        # Names neither the password nor the user, both of which are part of
        # the credential.
        description="credential embedded in a URL",
    )


def _assignment(line: str, number: int) -> SecretFinding | None:
    # Every assignment on the line, not the first: one line can hold a
    # reference and a literal, and rejecting the reference must not end the
    # search before the literal is judged.
    for match in _ASSIGNMENT.finditer(line):
        if _looks_generated(match.group("value")):
            return SecretFinding(
                rule="high_entropy_assignment",
                line=number,
                description=f"high-entropy value assigned to {match.group('key')}",
            )
    return None


def scan(text: str, *, max_findings: int = 5) -> list[SecretFinding]:
    """Findings describe what was seen; they never carry the value."""
    findings: list[SecretFinding] = []

    for number, line in enumerate(text.splitlines(), start=1):
        # One finding per line, most specific rule first: a line matching a
        # known token shape is already accounted for, and reporting it twice
        # would spend the cap on one line.
        finding = (
            _signature(line, number) or _url_credential(line, number) or _assignment(line, number)
        )
        if finding is not None:
            findings.append(finding)
        if len(findings) >= max_findings:
            break

    return findings
