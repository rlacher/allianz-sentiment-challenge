# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Defines schemas for the Feddit Sentiment API."""
from enum import Enum

from fastapi import Query, HTTPException, status
from pydantic import BaseModel, Field, field_validator


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
    time_from: int | None = Field(
        default=None,
        description="Optional UNIX timestamp to filter comments from a "
        "specific time (inclusive)"
    )
    time_to: int | None = Field(
        default=None,
        description="Optional UNIX timestamp to filter comments up to a"
        "specific time (exclusive)."
    )

    @field_validator("time_to")
    @classmethod
    def validate_time_range(cls, time_to, info):
        """Ensure that time_to is greater than or equal to time_from.

        Args:
            time_to: The end of the time range.
            info: Context information containing the data being validated,
            including time_from.
        Returns:
            The validated time_to value.
        Raises:
            HTTPException: If time_to is less than time_from.
        """
        time_from = info.data["time_from"]
        if (
            time_from is not None
            and time_to is not None
            and time_to < time_from
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid time range: time_to must be greater than or"
                "equal to time_from."
            )
        return time_to

    def __str__(self) -> str:
        """Return a string representation of the query parameters.

        Returns:
            A string summarizing the query parameters.
        """
        return (
            f"subfeddit_title={self.subfeddit_title}, "
            f"sort_by_polarity={self.polarity_sort_order}, "
            f"time_from={self.time_from}, "
            f"time_to={self.time_to}"
        )
