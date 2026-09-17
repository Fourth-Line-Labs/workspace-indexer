"""Relative, framework and package imports in one file."""

import json

import requests

from .models import Thing


class Store:
    def find(self, identifier: str) -> Thing:
        return Thing(identifier)

    def dump(self, thing: Thing) -> str:
        return json.dumps({"id": thing.identifier})

    def fetch(self, url: str) -> int:
        return requests.get(url).status_code
