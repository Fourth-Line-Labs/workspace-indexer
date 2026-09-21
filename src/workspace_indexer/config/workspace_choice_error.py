"""Raised when a caller must name a workspace and has not, or named a wrong one."""

from __future__ import annotations


class WorkspaceChoiceError(ValueError):
    """The same shape as `WorktreeChoiceError`, for the same reason.

    Neither default is safe once a config holds more than one workspace.
    Picking the first would answer from whichever happens to be listed first.
    Searching all of them would return one client's code to a question about
    another's, which is the exact thing separate workspaces exist to prevent --
    and it would do it silently, which is worse than an error.

    So one round trip, once, and the caller knows for the rest of the session.
    """

    def __init__(self, given: str | None, available: list[str]) -> None:
        self.given = given
        self.available = available
        listed = ", ".join(available) if available else "none are configured"
        if given is None:
            super().__init__(
                f"this config holds several workspaces ({listed}), which are indexed "
                "separately and do not share results. Name the one you mean."
            )
        else:
            super().__init__(f"unknown workspace {given!r}. Configured: {listed}.")
