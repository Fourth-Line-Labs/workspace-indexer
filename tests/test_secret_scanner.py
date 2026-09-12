"""Keeping credentials out of the index.

Indexed content is sent to an embedding provider as request input, so anything
reaching a chunk reaches a third party. Every token below is synthetic.

The two properties that matter: real credentials are caught whatever the file
is called, and the value never leaves the scanner.
"""

from __future__ import annotations

import pytest

from workspace_indexer.secrets import SecretFinding, scan, shannon_entropy

# Every fixture is assembled at runtime rather than written as a literal.
#
# Not decoration: the first attempt to push this file was rejected by GitHub's
# own push protection, which recognised the Slack-shaped fixture. Both scanners
# were right. A test suite for a secret detector cannot contain contiguous
# strings that look like secrets, so the prefixes are joined here and no
# scanner reading the source sees one.
_PREFIX = {
    "github": "github" + "_pat_",
    "aws": "AK" + "IA",
    "openai": "sk" + "-proj_",
    "slack": "xo" + "xb-",
    "google": "AI" + "za",
}

FAKE_GITHUB_PAT = _PREFIX["github"] + "11ABCDEFG0" + "z" * 30 + "Qw7"
FAKE_AWS = _PREFIX["aws"] + "IOSFODNN7EXAMPLE"
FAKE_OPENAI = _PREFIX["openai"] + "a1B2c3D4e5F6g7H8i9J0" * 2
FAKE_SLACK = _PREFIX["slack"] + "1234567890-abcdefghijklmnop"
FAKE_GOOGLE = _PREFIX["google"] + "B" * 35
FAKE_PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----"


def test_the_case_that_actually_bit_this_project() -> None:
    """.mcp.json held a GitHub token under a name no deny-list would flag.
    That is why the check is on content rather than filenames."""
    text = f'{{"headers": {{"Authorization": "Bearer {FAKE_GITHUB_PAT}"}}}}'
    findings = scan(text)
    assert findings
    assert findings[0].rule == "github_pat"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"key = {FAKE_AWS}", "aws_access_key"),
        (f"OPENAI={FAKE_OPENAI}", "openai"),
        (FAKE_PRIVATE_KEY, "private_key"),
        (f"token: {FAKE_SLACK}", "slack"),
        (f"k = {FAKE_GOOGLE}", "google_api"),
    ],
)
def test_known_token_shapes(text: str, rule: str) -> None:
    findings = scan(text)
    assert findings and findings[0].rule == rule


def test_a_secret_in_source_code_is_caught_too() -> None:
    """Not only config. A hard-coded key in a .py is the same problem."""
    code = f'STRIPE_KEY = "{FAKE_OPENAI}"\n\ndef charge():\n    pass\n'
    assert scan(code)


def test_high_entropy_assignment_to_a_credential_name() -> None:
    assert scan('DATABASE_PASSWORD = "k7Fq2mZx9RtVwLpA3nBcYdQe"')


# ---- what must NOT be blocked -----------------------------------------


def test_a_template_with_blank_values_is_fine() -> None:
    """.env.example is genuinely useful to index and holds nothing."""
    text = "VOYAGE_API_KEY=\nRERANK_MODEL=voyageai:rerank-2.5-lite\nQDRANT_MODE=embedded"
    assert scan(text) == []


def test_placeholders_are_not_secrets() -> None:
    for value in ("your-api-key-here", "changeme", "<YOUR_TOKEN>", "${API_KEY}", "TODO"):
        assert scan(f'API_KEY = "{value}"') == [], value


def test_prose_about_credentials_is_not_a_credential() -> None:
    """Documentation discussing tokens must stay indexable, or the guidance
    documents this project exists to find get withheld."""
    text = (
        "Authentication verifies the bearer token on every request. The "
        "API_KEY environment variable must be set before the service starts."
    )
    assert scan(text) == []


def test_a_url_is_not_a_secret() -> None:
    assert scan('AUTH_URL = "https://example.com/oauth/authorize/v2"') == []


def test_a_dotted_identifier_is_not_a_secret() -> None:
    assert scan('AUTH_BACKEND = "django.contrib.auth.backends.ModelBackend"') == []


def test_ordinary_config_values_pass() -> None:
    text = "EMBEDDING_MODEL=voyageai:voyage-code-4\nEMBEDDING_DIMENSIONS=1024\nLOG_LEVEL=INFO"
    assert scan(text) == []


# ---- the finding must not carry the value -----------------------------


def test_findings_never_contain_the_secret() -> None:
    """The finding is logged. A log line carrying the credential would defeat
    the entire exercise, and logs get shipped and shared like anything else."""
    findings = scan(f"token = {FAKE_GITHUB_PAT}")
    assert findings
    rendered = " ".join(str(f) + f.model_dump_json() for f in findings)
    assert FAKE_GITHUB_PAT not in rendered
    assert FAKE_GITHUB_PAT[:24] not in rendered


def test_finding_reports_a_usable_location() -> None:
    text = f"line one\nline two\ntoken = {FAKE_AWS}\n"
    findings = scan(text)
    assert findings[0].line == 3


def test_findings_are_capped() -> None:
    """A file of a thousand keys should not produce a thousand log lines."""
    text = "\n".join(f"KEY_{i} = {FAKE_AWS}" for i in range(50))
    assert len(scan(text)) <= 5


# ---- entropy ----------------------------------------------------------


def test_entropy_separates_random_from_english() -> None:
    assert shannon_entropy("k7Fq2mZx9RtVwLpA3nBcYdQe") > shannon_entropy("the quick brown fox")


def test_entropy_of_empty_string() -> None:
    assert shannon_entropy("") == 0.0


def test_finding_renders_without_a_value() -> None:
    finding = SecretFinding(rule="github_pat", line=7, description="GitHub token")
    assert "line 7" in str(finding)
    assert "github_pat" in str(finding)


def test_code_referencing_a_credential_is_not_a_credential() -> None:
    """`api_key=settings.qdrant_api_key` is a variable reference. This scanner
    withheld one of our own source files for exactly this before the fix."""
    assert scan("AsyncQdrantClient(url=settings.url, api_key=settings.qdrant_api_key)") == []
    assert scan("token = self._config.auth_token") == []
    assert scan("API_KEY = os.environ.get") == []


def test_an_actual_literal_next_to_a_reference_is_still_caught() -> None:
    """Ignoring identifiers must not become a way to hide a real value."""
    assert scan('client(api_key=settings.key, secret="k7Fq2mZx9RtVwLpA3nBcYdQe")')


# --- shapes found missing against a real repository ------------------------
#
# Values below are placeholders assembled to have the *shape* of a credential.
# The scanner keys on shape and entropy, never on a real value, so these
# exercise the same paths without committing a secret to a public repo.

_HIGH_ENTROPY = "Tl8QaFxOkB9J0aoe1qYaUdNbUkDCHDrDKYaMbtS"
_WITH_TILDE = "Tl8Q~FxOkB9J0aoe1qYaUdNbUkDCHDrDKYaMbtS"


def test_a_quoted_json_key_is_matched() -> None:
    """The single commonest way a credential is written down, and it was
    missed entirely: the pattern ran the key straight into `[:=]`, allowing no
    closing quote."""
    assert scan(f'"password": "{_HIGH_ENTROPY}"')


def test_a_value_containing_a_tilde_is_matched() -> None:
    """`~` was absent from the value character class, so an Azure service
    principal password failed to match at all rather than failing entropy."""
    assert scan(f'"password": "{_WITH_TILDE}"')
    assert scan(f'PASSWORD="{_WITH_TILDE}"')


@pytest.mark.parametrize(
    "line",
    [
        '"client_secret": "{v}"',
        '"apiKey": "{v}"',
        '$password = "{v}"',
        'ConnectionString="Server=x;Password={v};"',
    ],
)
def test_credential_shapes_across_languages(line: str) -> None:
    assert scan(line.format(v=_WITH_TILDE))


# --- what widening the pattern must not start catching ---------------------


@pytest.mark.parametrize(
    "line",
    [
        # Expressions, not literals. Withholding a whole file over a variable
        # reference is a silent loss of content -- the worse error here.
        'appConfigurationConnectionString = builder.Configuration["AppConfig"]',
        "instrumentationConnectionString: appInsights.properties.ConnectionString",
        "administratorLoginPassword: sqlAdminPassword",
        # camelCase and PascalCase identifiers clear the entropy bar easily.
        "ClientCredentials: ClientCredentialsSettingsValue",
        "api_key = settings.voyage_api_key",
        # Templates and placeholders.
        "password: ${DB_PASSWORD}",
        '"apiKey": "<YOUR_KEY_HERE>"',
        "VOYAGE_API_KEY=your-api-key-here",
    ],
)
def test_identifiers_and_expressions_are_not_secrets(line: str) -> None:
    assert not scan(line)


def test_an_all_letter_value_is_an_identifier_however_long() -> None:
    """A generated credential mixes letters with digits or symbols. All
    letters is a name, whatever its length or casing."""
    assert not scan('"password": "SomeVeryLongCamelCasedIdentifierName"')
    assert scan('"password": "SomeVeryLongCamelCased1dentifierN4me~x"')


# --- false positives found by indexing a real workspace --------------------
#
# Widening the value class to catch `~` made the entropy rule fire on ordinary
# code. Five files were purged from a live index over these before it was
# caught, which is a silent loss of content -- the worse failure of the two.


@pytest.mark.parametrize(
    "line",
    [
        # Rust and C++ paths.
        "auth_mode: AuthMode::TrustedLocal,",
        "config.auth_mode = ralph_api::AuthMode::TrustedLocal;",
        # A generic type.
        "pub auth: Option<AuthMeta>,",
        # The *name* of a credential, not the credential.
        "DOCKER_CREDENTIALS = 'docker-hub-credentials'",
        'api_key_name: "my-service-account-key"',
    ],
)
def test_paths_types_and_credential_names_are_not_secrets(line: str) -> None:
    assert not scan(line)


def test_hyphenated_and_underscored_names_read_as_identifiers() -> None:
    """`docker-hub-credentials` is all letters once separators are stripped.
    A generated value still mixes in digits or symbols."""
    assert not scan('"password": "some-long-kebab-cased-name-here"')
    assert not scan('"password": "some_long_snake_cased_name_here"')
    assert scan('"password": "some-long-kebab-c4sed-name~here"')


# --- expressions in C#, and credentials inside URLs ------------------------
#
# Both halves of #89. The first is a false positive that purged eight tracked
# files from a live index; the second is a false negative that would have sent
# a working credential to the embedding provider. Every value here is
# synthetic.


@pytest.mark.parametrize(
    "line",
    [
        # The exact shape that purged the eight files: the key matches on
        # `token`, and the value is an expression the scanner could not see
        # through because `?.` puts a `?` where it expected a dot.
        "    MaxOutputTokens = options?.MaxOutputTokens ?? _maxTokens,",
        "    ClaudeSecretRef = domain?.ClaudeSecretRef,",
        "    ApiKey = settings?.ApiKey ?? DefaultApiKeyValue,",
        "var token = context?.Request?.Headers?.Authorization;",
        # `??` alone, with no member access at all.
        "    ConnectionString = configured ?? FallbackConnectionValue,",
    ],
)
def test_csharp_null_conditional_expressions_are_not_secrets(line: str) -> None:
    assert not scan(line)


def test_a_real_literal_beside_a_null_conditional_is_still_caught() -> None:
    """Teaching the scanner about `?.` must not become a way to smuggle a
    value past it.

    This line used to be flagged for the wrong reason -- `options?.ApiKey`
    read as a generated value -- which happened to cover the literal after the
    `??`. Seeing through the expression would have uncovered it, so `??` is
    now an assignment operator in its own right: it is one.
    """
    assert scan(f'    ApiKey = options?.ApiKey ?? "{_WITH_TILDE}",')

    # Known gap, stated rather than asserted: the key is read from the token
    # immediately before the operator, so a fallback whose left side is not
    # credential-shaped -- `ApiKey = configured ?? "<literal>"` -- is not seen
    # by this rule. A provider-issued key there still trips `_SIGNATURES`.


def test_a_reference_earlier_on_the_line_does_not_end_the_search() -> None:
    """One line can hold both a reference and a literal. Rejecting the first
    must not stop the second from being judged -- otherwise writing a harmless
    assignment ahead of a real one hides it."""
    assert scan(f'client(api_key=settings.voyage_api_key, secret="{_HIGH_ENTROPY}")')
    assert scan(f'{{"auth": os.environ["X"], "password": "{_WITH_TILDE}"}}')


def test_a_credential_in_a_url_authority_is_caught() -> None:
    """No assignment rule can see this shape -- there is no `key = value`.
    Before this rule the credential shipped to the embedding provider unless
    an unrelated query parameter happened to trip the entropy check."""
    findings = scan("mongodb://admin:devpassword123@localhost:27017/")
    assert findings and findings[0].rule == "url_credential"


def test_the_url_rule_does_not_depend_on_a_query_string() -> None:
    """The case that prompted this was found *with* a query string and was
    flagged for the wrong reason -- an unrelated `authSource` parameter. Both
    forms must be caught by the URL rule itself."""
    bare = "postgres://svc:Xk29fbQ2wwTmeeQ@db.internal:5432/app"
    assert [f.rule for f in scan(bare)] == ["url_credential"]
    assert [f.rule for f in scan(bare + "?sslmode=require&pool=10")] == ["url_credential"]


def test_a_weak_password_in_a_url_is_still_a_password() -> None:
    """Entropy is not consulted for this shape. `user:value@host` is written
    for one reason, and a weak credential is still a credential."""
    assert scan("redis://cache:hunter2xyz@10.0.0.4:6379/0")


def test_the_finding_names_neither_the_user_nor_the_password() -> None:
    findings = scan("mongodb://admin:devpassword123@localhost:27017/")
    rendered = " ".join(str(f) + f.model_dump_json() for f in findings)
    assert "devpassword123" not in rendered
    assert "admin" not in rendered


@pytest.mark.parametrize(
    "line",
    [
        # Documentation showing the syntax is exactly what this index is for.
        "postgres://user:password@host:5432/dbname",
        "mongodb://USER:PASSWORD@cluster.example.com/",
        "postgres://user:<password>@host/db",
        "postgres://user:${DB_PASSWORD}@host/db",
        "mongodb://admin:changeme@localhost:27017/",
        "postgres://user:****@host/db",
        # No password at all.
        "https://example.com/oauth/authorize/v2",
        "redis://localhost:6379/0",
        "git@github.com:Fourth-Line-Labs/workspace-indexer.git",
    ],
)
def test_url_syntax_examples_are_not_withheld(line: str) -> None:
    assert not scan(line)


def test_a_query_parameter_value_ends_at_the_ampersand() -> None:
    """`authSource=admin` names a database. It read as a credential because
    the value ran on through the rest of the query string, which cleared the
    entropy bar -- so a file was withheld over a parameter, while the real
    credential in the same URL went unnoticed."""
    assert scan("?authSource=admin&directConnection=true&serverSelectionTimeoutMS=10000") == []
    assert scan("connect?authSource=admin&retryWrites=true&w=majority") == []


def test_a_credential_in_a_query_parameter_is_still_caught() -> None:
    """Ending the value at `&` must not stop the parameter's own value from
    being judged."""
    assert scan(f"https://api.example.com/v1/items?api_key={_HIGH_ENTROPY}&page=2")
