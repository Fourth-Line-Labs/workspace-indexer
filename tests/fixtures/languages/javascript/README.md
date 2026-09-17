# JavaScript fixtures

| file | what it is here for |
|---|---|
| `lib/util.js` | the target |
| `lib/index.js` | a re-export, so the directory has an entry point |
| `app.js` | a path through the index file, a Node builtin that is only a builtin *with* the `node:` prefix (`node:path`), and a package (`lodash`) |

`node:path` matters: `path` without the prefix is a builtin too, but several of
Node's newer ones — `test`, `sqlite`, `sea` — are builtins **only** with it, and
the prefix is what tells a framework module from someone's local package.
