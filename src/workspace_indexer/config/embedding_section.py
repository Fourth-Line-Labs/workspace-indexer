"""Per-workspace overrides of the embedding settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from workspace_indexer.config.strict import Strict

if TYPE_CHECKING:
    from workspace_indexer.config.settings import Settings


class EmbeddingSection(Strict):
    """What one workspace embeds with, where that differs from the default.

    Embedding settings live in `.env` because that is where they started, next
    to the API key. They are not secrets -- a model name and a dimension count
    are the plainest possible "what to index" facts -- and putting them in
    `workspace.yaml` is what lets two workspaces in one process differ.

    The motivating case is not tuning. On a machine holding work for more than
    one client, one client's code may be barred from reaching a hosted API at
    all, so that workspace embeds with a local model while another does not.
    A process-wide model cannot express that.

    Every field is optional and None means "inherit". A workspace that says
    nothing about embedding gets exactly what `.env` gives it, which is what
    every existing config does.
    """

    model: str | None = None
    dimensions: int | None = None
    quantization: Literal["float32", "int8", "binary"] | None = None
    sparse_model: str | None = None
    # Follows the model, or the cost report prices one workspace's tokens at
    # another's rate. Nothing else notices, because the number is only ever
    # used where the provider reports no price of its own.
    price_per_mtok: float | None = None

    @model_validator(mode="after")
    def _a_model_carries_its_width(self) -> EmbeddingSection:
        """Naming a model without its dimensions is the easy mistake here.

        Nothing cross-validates the pair: the space would claim whatever
        `.env` says while the backend returns something else, and the mismatch
        surfaces at the first embed batch -- after indexing has started and
        spent tokens, rather than at config load. The old `.env`-only
        arrangement at least kept the two lines next to each other; per-field
        inheritance makes omission the easy path, so it is refused.
        """
        if self.model is not None and self.dimensions is None:
            raise ValueError(
                f"embedding model {self.model!r} is set without `dimensions:`. The two "
                "belong together -- inheriting the dimension count from .env while "
                "overriding the model gives a space whose width does not match the "
                "vectors, and that fails at the first embed batch rather than here."
            )
        return self

    def applied_to(self, settings: Settings) -> Settings:
        """`settings`, with this workspace's overrides in place.

        Returned as a whole `Settings` rather than carried separately so that
        everything derived from it moves together -- `config_hash` most of all,
        since it covers the model, the dimensions, the quantization and the
        sparse model. A hash that did not follow the override would report two
        workspaces on different models as comparable runs.

        Re-validated rather than copied in: `model_copy(update=...)` does not
        run field validators, so an override would reach the embedder without
        ever being checked.
        """
        overrides = {
            f"embedding_{name}": value
            for name, value in (
                ("model", self.model),
                ("dimensions", self.dimensions),
                ("quantization", self.quantization),
                ("price_per_mtok", self.price_per_mtok),
            )
            if value is not None
        }
        if self.sparse_model is not None:
            overrides["sparse_model"] = self.sparse_model
        if not overrides:
            return settings
        # The keys are assembled by convention, with one name that breaks it.
        # `Settings` ignores extras, so a key that stopped matching would be
        # dropped by the round trip with no error and no effect -- the same
        # silent shape as a documented setting nothing reads, which this
        # codebase has shipped before.
        unknown = sorted(set(overrides) - set(type(settings).model_fields))
        if unknown:
            raise ValueError(f"no such settings: {unknown}; the override mapping is stale")
        return type(settings).model_validate({**settings.model_dump(), **overrides})
