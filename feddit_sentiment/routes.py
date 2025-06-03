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
        params: Query parameters including subfeddit title and optional
        polarity sort order.
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
                COMMENT_LIMIT,
                params.polarity_sort_order
            )
    except ValueError as value_error:
        logger.warning(
            f"Enriching comments failed in service layer with: {value_error}"
        )
        raise HTTPException(status_code=404, detail=str(value_error))

    return _format_output(
        subfeddit_id,
        params.subfeddit_title,
        params.polarity_sort_order,
        enriched_comments
    )


def _format_output(
        subfeddit_id: int,
        subfeddit_title: str,
        polarity_sort: SortOrder | None,
        enriched_comments: list[dict]
) -> dict:
    """Maps meta and comment data into output dictionary.

    Args:
        subfeddit_id: The subfeddit's unique id.
        subfeddit_title: The subfeddit's title.
        polarity_sort: Sort order for polarity scores or None.
        enriched_comments: A list of comments, each with a polarity score
        and sentiment label.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        TypeError: If any of the input parameters are of the wrong type.
    """
    if not isinstance(subfeddit_id, int):
        raise TypeError("Subfeddit ID must be of type int")
    if not isinstance(subfeddit_title, str):
        raise TypeError("Subfeddit title must be of type string")
    if not isinstance(polarity_sort, (SortOrder, type(None))):
        raise TypeError("Polarity sort must be of type SortOrder or None")
    if not isinstance(enriched_comments, list):
        raise TypeError("Enriched comments must be of type list")

    return {
        "subfeddit": {
            "id": subfeddit_id,
            "title": subfeddit_title
        },
        "comment_count": len(enriched_comments),
        "sort": {
            "key": "polarity" if polarity_sort else "created_at",
            "order": "asc" if polarity_sort == SortOrder.asc else "desc"
        },
        "comments": enriched_comments
    }
