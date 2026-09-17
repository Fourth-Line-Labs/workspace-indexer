// TypeScript's ESM convention: the specifier names the *emitted* file, so a
// literal lookup for `models.js` finds nothing and the resolver has to try
// `models.ts`.
import { Thing } from './models.js';

export const empty: Thing = { id: '' };
