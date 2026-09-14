"""Turning a file's bytes into the text every reader of it sees.

One function rather than a convention, because there are two paths to the same
file -- the indexing walk and the graph backfill -- and they have to produce
identical text. A grammar handed different bytes depending on which path
reached the file is a difference nothing downstream can see until an
extraction quietly disagrees with itself.
"""

from __future__ import annotations


def decode_source(raw: bytes) -> str:
    """`utf-8-sig`, strict, with line endings left alone.

    `utf-8-sig` rather than `utf-8`: it strips a leading byte-order mark and is
    otherwise identical. Visual Studio writes one on almost everything -- 376
    of 479 .cs files in one real workspace -- and a stray U+FEFF is not
    whitespace to a regex, so every line-anchored pattern silently fails on the
    first line. That cost the Razor route scanner three quarters of its matches
    before anyone noticed the character was there.

    Decoding the bytes rather than opening in text mode, because text mode
    translates CRLF to LF and Visual Studio writes those too. The difference
    does not change what the C# grammar extracts, which is exactly why it would
    go unnoticed: this is one decoder so the question cannot arise.
    """
    return raw.decode("utf-8-sig")
