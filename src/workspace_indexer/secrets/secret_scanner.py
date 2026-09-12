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
# That only reaches the fallback when the token before the `??` is itself
# credential-shaped. `ApiKey = configured ?? "literal"` is the same line with
# the name on the other side of the `=`, and `_FALLBACK` below is what carries
# the key across.
#
# A query string is a run of assignments rather than one value:
# `?authSource=admin&directConnection=true&...` was reading everything after
# `authSource=` as that parameter's value, which cleared the entropy bar and
# withheld a file over a parameter naming a database. The credential in that
# same URL sits in the authority and is caught by `_URL_CREDENTIAL` below.
#
# The value is cut at `&` by `_first_parameter` rather than by excluding `&`
# from the class. Excluding it looks equivalent and is not: `&` is a legal
# password character, so a generated value with one in it truncates at the
# ampersand, falls under the sixteen-character floor, and is then missed
# *entirely* rather than judged and cleared -- a false positive traded for a
# false negative, which is the worse of the two. The test named for that shape
# holds the example; writing it out here made this file's own rule fire on the
# file, which is the ordinary way a comment becomes a credential.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.-]*
        (?:api[_-]?key|secret|token|password|passwd|credential|auth|
           connection[_-]?string|conn[_-]?str|sas|pwd)
        [A-Za-z0-9_.-]*)
    ["']?              # a quoted key: "password": "..."
    \s* (?: [:=] | \?\? ) \s*   # `??` assigns a default in C#, the same way
    ["']?
    (?P<value>[^\s"'`,;)\]}]{16,})
    """
)

# A default that follows a name this rule already cares about:
# `ApiKey = configured ?? "<literal>"`. `_ASSIGNMENT` matches key-then-value,
# so it reads `configured` here and stops; the literal that actually becomes
# `ApiKey` sits one operator further along.
#
# The key is required to be credential-shaped and the fallback is required to
# be quoted, which is what keeps this from claiming every `??` on a line. The
# value is judged by the same entropy rules as any other.
_FALLBACK = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.-]*
        (?:api[_-]?key|secret|token|password|passwd|credential|auth|
           connection[_-]?string|conn[_-]?str|sas|pwd)
        [A-Za-z0-9_.-]*)
    ["']? \s* [:=] [^\n]*? \?\? \s*
    ["'] (?P<value>[^"'\n]{16,}) ["']
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
#
# A host is required after the `@`, not just the `@`, or an unfinished
# documentation example with the host elided reads as a live credential and
# takes the whole page out of the index. The tests hold the example: written
# out here, the word after it becomes the host and the rule fires on its own
# explanation.
#
# The password may contain `:`; only the user may not. RFC 3986 allows colons
# after the first one in userinfo, so a generated password containing one was
# invisible to this rule. The trailing `@` still anchors the match, so widening
# the class cannot make it run away.
_URL_CREDENTIAL = re.compile(
    r"(?i)\b[a-z][a-z0-9+.\-]*://"
    r"(?P<user>[^\s:/?#@\[\]]+):(?P<password>[^\s/?#@\[\]]+)@"
    r"(?P<host>\[[0-9A-Fa-f:.]+\]|[^\s/?#@\[\]]+)"
)

# Passwords that name the *idea* of a password rather than being one.
# Documentation showing connection-string syntax is exactly the content this
# index exists to retrieve, so withholding a page over `user:password@host`
# would be the destructive error. Compared both as written and with separators
# stripped, so `Your_Password` and `your-password` are both covered.
#
# `admin` and `root` are deliberately *not* here, and neither is `user`. They
# read like placeholders and are not: they are the most commonly deployed
# default credentials there are, so a root-as-password connection string
# against an internal address is a real finding. The rule this set serves is
# that a weak password is still a password, and an entry that is also a
# plausible literal value breaks it.
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


# `%NAME%` -- a Windows environment variable, not a password that happens to
# begin with a percent-encoded character.
_WINDOWS_VARIABLE = re.compile(r"%[A-Za-z0-9_]+%")

# Shortest run of one repeated character that reads as masking rather than as
# a value. Four is short enough to catch `xxxx` and long enough that a
# two-character password is judged on its merits.
_MASK_LENGTH = 4


# The start of the *next* query parameter: an `&` with a `key=` behind it.
# What separates a query string from a password containing `&` is what follows
# the ampersand, so that is what this looks at.
_NEXT_PARAMETER = re.compile(r"&(?=[A-Za-z0-9_.\[\]-]+=)")


def _first_parameter(value: str) -> str:
    """One parameter's value, where the value ran on into the next one.

    `authSource=admin&directConnection=true&serverSelectionTimeoutMS=10000`
    is three assignments, and reading it as one gave `authSource` a value with
    enough entropy to withhold the file -- over a parameter naming a database.

    Cut here rather than by excluding `&` from the value class, because `&` is
    a legal password character: excluding it truncated such a value at the
    ampersand, leaving a remainder under the length floor that was never judged
    at all. `test_a_password_containing_an_ampersand_is_still_seen_whole` holds
    the example -- spelled out here, it would withhold this file.
    """
    parameter = _NEXT_PARAMETER.split(value, maxsplit=1)
    return parameter[0]


def _is_placeholder_password(password: str) -> bool:
    """A sample, not a secret: `<password>`, `${PASSWORD}`, `%PW%`, `xxxxxxxx`."""
    if password.startswith(("<", "${", "{")):
        return True
    # `%NAME%`, both ends. A bare `%` prefix would have classified every
    # percent-encoded password as a placeholder, and percent-encoding is the
    # standard way (RFC 3986) to put a special character into userinfo -- so
    # `%40dmin12345%21` would have shipped to the provider as a sample.
    if _WINDOWS_VARIABLE.fullmatch(password):
        return True
    lowered = password.lower()
    # Both spellings, because the two sets are written differently: the URL set
    # holds bare words and `_PLACEHOLDERS` holds hyphenated ones like
    # `replace-me`. Comparing only the stripped form left every hyphenated
    # entry unreachable from here, so a docs URL using one was flagged.
    bare = lowered.translate(_WORD_SEPARATORS)
    if {lowered, bare} & (_URL_PLACEHOLDER_PASSWORDS | _PLACEHOLDERS):
        return True
    # A masking run, as a page redacting its own example. Length rather than
    # punctuation: `xxxxxxxx` and `aaaaaaaa` mask exactly as `********` does,
    # and treating them as credentials withholds the page they document.
    return len(password) >= _MASK_LENGTH and len(set(password)) == 1


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
    # Every URL on the line, for the same reason `_assignment` judges every
    # assignment: a documentation example written in front of a real
    # connection string would otherwise be the only thing looked at, and
    # nothing else sees a credential in an authority.
    for match in _URL_CREDENTIAL.finditer(line):
        if _is_placeholder_password(match.group("password")):
            continue
        return SecretFinding(
            rule="url_credential",
            line=number,
            # Names neither the password nor the user, both of which are part
            # of the credential.
            description="credential embedded in a URL",
        )
    return None


def _assignment(line: str, number: int) -> SecretFinding | None:
    # Every assignment on the line, not the first: one line can hold a
    # reference and a literal, and rejecting the reference must not end the
    # search before the literal is judged.
    for match in _ASSIGNMENT.finditer(line):
        if _looks_generated(_first_parameter(match.group("value"))):
            return SecretFinding(
                rule="high_entropy_assignment",
                line=number,
                description=f"high-entropy value assigned to {match.group('key')}",
            )
    for match in _FALLBACK.finditer(line):
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
