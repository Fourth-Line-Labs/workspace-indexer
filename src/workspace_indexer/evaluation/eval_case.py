"""One query and the files it should surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Retrieval cases ask "can the index find a thing at all" and are the
# regression net. Guidance cases ask the question an agent asks on greenfield
# work -- "how am I supposed to build this?" -- where the tempting wrong
# answers sit in the same directory with the same vocabulary. They are the
# document-classification argument, and they are scored separately because
# averaging them together hides exactly the movement worth seeing.
# Lexical cases are the control group for the grep comparison: queries whose
# answer is a token that appears verbatim in the file -- an identifier, an env
# var, an error string. They exist because the other two groups were written to
# defeat lexical matching, so scoring grep on those alone would prove only that
# the dataset was built to beat it. Where grep wins, this group is where it
# shows, and a comparison that cannot report a loss is not a measurement.
Group = Literal["retrieval", "guidance", "lexical"]


class EvalCase(BaseModel):
    query: str
    group: Group = "retrieval"
    # Substring match against rel_path, so a case survives a file moving
    # within a directory without needing the dataset rewritten.
    expect: list[str]
    # The pattern a competent engineer would reach for, written by hand and
    # written to win. Stored with the case so a reader can object that it was
    # too weak -- which is the only defence against a comparison quietly
    # steelmanning one side. Required when scoring a lexical tool; absent, the
    # run fails rather than falling back to the raw query, because the raw
    # query matches nothing and would look like an honest zero.
    grep_pattern: str | None = None
    note: str | None = None
