# TSX fixtures

A separate language from TypeScript as far as the extractor is concerned, so it
needs its own row: a `.tsx` file resolving a `.tsx` neighbour is not covered by
anything in `typescript/`.

| file | what it is here for |
|---|---|
| `Button.tsx` | the target |
| `App.tsx` | a relative first-party import and a package (`react`), in a file whose body is JSX rather than plain TypeScript |
