"""An absolute intra-package import, resolved by matching the module path
against an indexed file's suffix rather than by reproducing sys.path."""

from pkg.store import Store


class Service:
    def __init__(self) -> None:
        self.store = Store()
