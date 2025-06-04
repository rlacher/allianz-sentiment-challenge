# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Route layer of Feddit Sentiment API

Defines the API's /comments endpoint including JSON output formatting.
"""
import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status
)
from feddit_sentiment.config import API_VERSION
from feddit_sentiment import service
from feddit_sentiment.schemas import SortOrder, CommentQueryParams

BASE_PATH = f"/api/{API_VERSION}"
COMMENT_LIMIT = 25

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(f"{BASE_PATH}/comments")
def get_comments_sentiment(params: CommentQueryParams = Depends()) -> dict:
    """Fetch and analyse comments from the given subfeddit.

    Delegates to high-level service layer function `get_enriched_comments()`.

    Args:
        params: Query parameters including subfeddit title, optional
        polarity sort order and optional time range.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        HTTPException: If the subfeddit does not exist
        (implied by service layer).
    """
    logger.info(f"Received request parameters: {params}")

    try:
        enriched_comments, subfeddit_id = \
            service.get_enriched_comments(
                params.subfeddit_title,
                params.polarity_sort_order,
                params.time_from,
                params.time_to,
                COMMENT_LIMIT
            )
    except ValueError as value_error:
        logger.warning(
            f"Enriching comments failed in service layer with: {value_error}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(value_error)
        )

    return _format_output(
        subfeddit_id,
        params.subfeddit_title,
        params.polarity_sort_order,
        params.time_from,
        params.time_to,
        enriched_comments
    )


def _format_output(
        subfeddit_id: int,
        subfeddit_title: str,
        polarity_sort: SortOrder | None,
        time_from: int | None,
        time_to: int | None,
        enriched_comments: list[dict]
) -> dict:
    """Maps meta and comment data into output dictionary.

    Args:
        subfeddit_id: The subfeddit's unique id.
        subfeddit_title: The subfeddit's title.
        polarity_sort: Sort order for polarity scores or None.
        time_from: Optional UNIX timestamp to filter comments from a
        specific time.
        time_to: Optional UNIX timestamp to filter comments up to a
        specific time.
        enriched_comments: A list of comments, each with a polarity score
        and sentiment label.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        TypeError: If `enriched_comments` is not a list of
        dictionaries.
    """
    if not isinstance(enriched_comments, list):
        raise TypeError(
            "enriched_comments must be of type list"
        )
    elif not all(isinstance(c, dict) for c in enriched_comments):
        raise TypeError(
            "enriched_comments must be a list of dictionaries"
        )

    # Construct filter dictionary only if time filters are provided
    filter_info = None
    if time_from is not None or time_to is not None:
        filter_info = {}
        if time_from is not None:
            filter_info["time_from"] = time_from
        if time_to is not None:
            filter_info["time_to"] = time_to
    else:
        filter_info = "None"

    # Sort metadata
    sort_info = {
        "key": "polarity" if polarity_sort is not None else "created_at",
        "order": "asc" if polarity_sort == SortOrder.asc else "desc"
    }

    # Build final structured output
    output = {
        "subfeddit": {
            "id": subfeddit_id,
            "title": subfeddit_title
        },
        "comment_count": len(enriched_comments),
        "filter": filter_info,
        "sort": sort_info,
        "comments": enriched_comments
    }

    return output
