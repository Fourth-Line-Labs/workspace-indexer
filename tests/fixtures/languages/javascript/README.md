# JavaScript fixtures

| file | what it is here for |
|---|---|
| `lib/util.js` | the target |
| `lib/index.js` | a re-export, so the directory has an entry point |
| `app.js` | a directory specifier resolved through the index file (`./lib`), an ordinary builtin (`node:path`), a **prefix-only** builtin (`node:test`), and a package (`lodash`) |

Both builtins are here on purpose. `path` is a builtin with or without the
prefix, so it cannot exercise the rule that matters: `test`, `sqlite` and `sea`
are builtins **only** with `node:`, and without it `test` is someone's local
package. A corpus holding only `node:path` would keep passing after a
regression that stopped recognising the prefix-only ones.
