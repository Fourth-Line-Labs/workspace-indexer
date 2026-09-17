"""Values that look like credentials to a scanner and are not.

This file exists to be *indexed*. A false positive here is silent data loss --
the file disappears from the index and nothing says so -- which is why the
baseline records which files were withheld rather than only which were not.

Every line is a shape that has actually been flagged by some version of this
scanner, so each one rests on a different rule: an expression rather than a
literal, a dotted attribute, a template, a name made only of letters, a value
too short to clear the entropy floor, and a query parameter that is not the
credential in its own URL.
"""

MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 30

# A reference, not a value: the credential lives in settings.
API_KEY = settings.voyage_api_key
AUTH_TOKEN = self._config.auth_token

# Templates, which name a credential rather than holding one.
CONNECTION_STRING_TEMPLATE = "postgres://user:${DB_PASSWORD}@{host}/{database}"
PASSWORD_PLACEHOLDER = "<YOUR_PASSWORD_HERE>"
MUSTACHE_SECRET = "{{DB_PASSWORD}}"

# All letters once separators are stripped: the *name* of a credential.
DOCKER_CREDENTIALS_NAME = "docker-hub-credentials"
SQL_ADMIN_PASSWORD_KEY = "sqlAdminPassword"

# A dotted path and a generic type, neither of which is a literal.
AUTH_BACKEND = "django.contrib.auth.backends.ModelBackend"
AUTH_MODE = AuthMode.TrustedLocal

# A query parameter naming a database, in a URL whose authority holds no
# credential at all.
MONGO_OPTIONS = "?authSource=admin&directConnection=true&serverSelectionTimeoutMS=10000"
