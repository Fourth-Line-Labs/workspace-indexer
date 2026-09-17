# Python fixtures

Resolution here is by path: a relative import names a neighbour, and an
absolute one is matched against an indexed file's suffix rather than by
reproducing `sys.path`.

| file | what it is here for |
|---|---|
| `pkg/models.py` | no imports at all — zero is a real answer, and a file with none must not read as one that was never scanned |
| `pkg/store.py` | the three origins in one file: relative first-party (`.models`), framework (`json`), package (`requests`) |
| `pkg/service.py` | an absolute intra-package import (`pkg.store`), resolved by suffix match |
| `pkg/aliased.py` | `import os.path as osp`, `from os import sep as separator`, and `from . import models` — the alias is not the module, and the bare `.` resolves to the package `__init__` |
| `config_values.py` | **must be indexed.** A token limit, a connection-string template, a credential's *name* and a dotted backend path: everything a scanner overreacts to, and none of it a secret |
| `credentials.py` | **must be withheld.** Synthetic, no provider prefix — a real token shape would trip the host's own push protection, which this project has already found out once |
