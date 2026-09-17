"""Shapes the extractor has to spell out rather than take from the statement."""

import os.path as osp
from os import sep as separator

from . import models

PARENT = osp.dirname(__file__)
SEPARATOR = separator
THING = models.Thing
