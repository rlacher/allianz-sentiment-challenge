# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Defines schemas for the Feddit Sentiment API."""
from enum import Enum

from fastapi import Query
from pydantic import BaseModel


class SortOrder(str, Enum):
    "Enum to define a sort order."
    asc = "asc"
    desc = "desc"


class CommentQueryParams(BaseModel):
    """Query parameters for the /comments endpoint."""
    subfeddit_title: str
    polarity_sort_order: SortOrder | None = Query(
        default=None,
        description="Optional sort order for polarity scores (asc or desc)."
    )

    def __str__(self) -> str:
        return (
            f"subfeddit_title={self.subfeddit_title}, "
            f"sort_by_polarity={self.polarity_sort_order}"
        )
