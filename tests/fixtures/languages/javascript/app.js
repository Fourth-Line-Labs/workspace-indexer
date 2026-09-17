// `./lib` resolves through the directory's index file.
import { slug } from './lib/index.js';
// A Node builtin reachable only with the prefix, and a package.
import { join } from 'node:path';
import lodash from 'lodash';

export const name = slug(join('a', 'b'));
export { lodash };
