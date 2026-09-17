// A directory specifier, which resolves through the barrel's index file.
import { Store } from './barrel';
// A bare specifier: a package, unresolvable without node_modules.
import { debounce } from 'lodash';
// Node's own library, via the prefix that makes it unambiguous.
import { readFile } from 'node:fs/promises';

export const store = new Store();
export { debounce, readFile };
