"""One command-line search tool, as the eval harness drives it.

A value object rather than two retriever subclasses, because grep and ripgrep
differ only in which binary runs and which flags it takes -- the parsing, the
ranking and the failure handling are identical, and splitting them into a type
each would duplicate the part that can actually be wrong.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from workspace_indexer.evaluation.tool_missing_error import ToolMissingError

# A directory name has to reach the tools as a literal. Both `--exclude-dir`
# and ripgrep's `--glob` take a pattern, so a name carrying any of these would
# silently exclude the wrong directories -- or, with an unbalanced bracket,
# make ripgrep exit 2 and kill the run. There is no portable escape that means
# the same thing in both dialects, so an untranslatable name is refused.
_GLOB_METACHARACTERS = frozenset("*?[]")


@dataclass(frozen=True)
class LexicalTool:
    name: str
    binary: str
    flags: tuple[str, ...]
    # How this tool is told to skip a directory, and a file. The spellings
    # differ between the two binaries; the exclusions themselves come from the
    # index's own effective set rather than being written out twice.
    exclude_dir_template: str
    exclude_file_template: str

    def argv(
        self,
        pattern: str,
        root: Path,
        dirs: tuple[str, ...] = (),
        files: tuple[str, ...] = (),
    ) -> list[str]:
        """The command, resolved to an absolute path and never shelled out.

        `shutil.which` rather than the bare name because a shell function can
        shadow the binary: on the machine this was written, `grep` resolves to
        a wrapper around ugrep, so invoking it through a shell would measure a
        different tool and label the result `grep`.
        """
        resolved = shutil.which(self.binary)
        if resolved is None:
            raise ToolMissingError(f"{self.binary} is not on PATH")
        skips = [self.exclude_dir_template.format(name=name) for name in dirs]
        skips += [self.exclude_file_template.format(name=name) for name in files]
        return [resolved, *self.flags, *skips, "-e", pattern, str(root)]


# -c, so the output is bounded by the number of files rather than the number of
# matches: a pattern matching half the corpus would otherwise produce megabytes
# of lines to count, and the flood is one of the things being measured.
#
# -i and -E, because both make grep *stronger*: case-insensitivity and
# alternation are what a hand-written pattern needs to be a fair opponent.
GREP = LexicalTool(
    name="grep",
    binary="grep",
    flags=("-r", "-c", "-I", "-i", "-E"),
    exclude_dir_template="--exclude-dir={name}",
    # GNU grep matches --exclude against the basename only, so a path-shaped
    # exclusion reaches it less precisely than it reaches ripgrep. Noted rather
    # than hidden: it is one of several reasons ripgrep is the comparator to
    # read, and it can only ever exclude *more* than intended, never less.
    exclude_file_template="--exclude={name}",
)

# --hidden, because `Walker` indexes hidden files and directories deliberately
# -- `.claude/` is a primary target, not something to skip -- while ripgrep
# skips them by default and grep does not. Measured before it was fixed: over
# the 24-case dataset grep returned ten hits under hidden directories and
# ripgrep returned none, and their total breadth differed by 109 files. Two
# tools scoring the same while reading different corpora is the failure this
# comparison is least able to detect from its own output.
#
# Only hidden files; gitignore handling stays as it is. That difference is
# real, is one of the things the comparison is for, and is reported rather
# than flattened.
RIPGREP = LexicalTool(
    name="ripgrep",
    binary="rg",
    flags=("-c", "-i", "--hidden"),
    exclude_dir_template="--glob=!{name}/",
    exclude_file_template="--glob=!{name}",
)


def translate_excludes(patterns: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The index's effective exclusions, as directory names and file globs.

    This is a correctness boundary, not a convenience. `HARDCODED_EXCLUDES`
    exists because an eval record or the dataset quotes every query and every
    pattern verbatim, which makes it a near-perfect match for the queries being
    scored; the index refuses to index them for that reason. A lexical tool not
    given the same exclusions searches the harness's own output and scores it,
    which is how a comparison ends up measuring itself.

    `name/**`, `**/name/**` and `name/` all become a directory; anything else
    contributes its last segment as a file glob. Only the final segment
    survives, so a rooted pattern like `**/data/qdrant/**` excludes any
    `qdrant` directory rather than only that one. Over-exclusion, and
    deliberately the chosen direction: searching *more* than the index is what
    invalidates a run.

    The trailing-slash spelling is a directory to `IgnoreMatcher` and is how a
    user writes one in `index.exclude`. Taking its last segment as a file glob
    produced the empty string, which excluded nothing and left the tools
    searching a directory the index does not have.
    """
    dirs: list[str] = []
    files: list[str] = []
    for pattern in patterns:
        trimmed = pattern.removeprefix("**/")
        directory = None
        if trimmed.endswith("/**"):
            directory = trimmed[: -len("/**")]
        elif trimmed.endswith("/"):
            directory = trimmed.rstrip("/")
        if directory is not None:
            name = directory.split("/")[-1]
            if set(name) & _GLOB_METACHARACTERS:
                raise ValueError(
                    f"exclusion {pattern!r} names a directory with glob characters, which "
                    "cannot be passed to grep and ripgrep as the same literal"
                )
            if name:
                dirs.append(name)
            continue
        name = trimmed.split("/")[-1]
        # An empty glob is not a no-op: it reaches the tools as `--exclude=`,
        # which is a pattern matching nothing and reads in the argv as though
        # something had been excluded.
        if name:
            files.append(name)
    return tuple(dict.fromkeys(dirs)), tuple(dict.fromkeys(files))
