# TypeScript fixtures

The JS family resolves relative specifiers by trying extensions and index
files, and declines bare ones — a package needs `node_modules`, which is not a
guess worth making.

| file | what it is here for |
|---|---|
| `src/models.ts` | the target, with no imports of its own |
| `src/store.ts` | an extension-less relative specifier |
| `src/emitted.ts` | `./models.js` — TypeScript's ESM convention, where the specifier names the *emitted* file and a literal lookup finds nothing |
| `src/barrel/index.ts` | a re-export, which makes a directory specifier resolvable |
| `src/consumer.ts` | a directory specifier (`./barrel`), a package (`lodash`, unclassified), and a Node builtin reachable only with the prefix (`node:fs/promises`) |
