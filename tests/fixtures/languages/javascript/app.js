// `./lib` with no file name: resolution has to try the directory's index file,
// which is a different path through the resolver than naming the file.
import { slug } from './lib';
// A builtin either way, and one that is a builtin *only* with the prefix --
// without it, `test` is someone's local package. Both are here because a
// regression in the prefix-only rule would leave `node:path` classified
// correctly and pass a corpus that held only that.
import { join } from 'node:path';
import { test } from 'node:test';
// A package, unresolvable without node_modules.
import lodash from 'lodash';

export const name = slug(join('a', 'b'));
export { lodash, test };
