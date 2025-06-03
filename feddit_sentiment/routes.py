# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Route layer of Feddit Sentiment API

Defines the API's /comments endpoint including JSON output formatting.
"""
import logging

from fastapi import APIRouter, HTTPException

from feddit_sentiment.config import API_VERSION
from feddit_sentiment import service

BASE_PATH = f"/api/{API_VERSION}"
COMMENT_LIMIT = 25

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(f"{BASE_PATH}/comments")
def get_comments_sentiment(subfeddit_title: str) -> dict:
    """Fetch and analyse comments from the given subfeddit.

    Delegates to high-level service layer function `get_enriched_comments()`.

    Args:
        subfeddit_title: The subfeddit's title.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        HTTPException: If the subfeddit does not exist
        (implied by service layer).
    """
    try:
        enriched_comments, subfeddit_id = \
            service.get_enriched_comments(subfeddit_title, COMMENT_LIMIT)
    except ValueError as value_error:
        logger.warning(
            f"Enriching comments failed in service layer with: {value_error}"
        )
        raise HTTPException(status_code=404, detail=str(value_error))

    return _format_output(
        subfeddit_id,
        subfeddit_title,
        enriched_comments
    )


def _format_output(
        subfeddit_id: int,
        subfeddit_title: str,
        enriched_comments: list[dict]
) -> dict:
    """Maps meta and comment data into output dictionary.

    Information regarding sorting is hardcoded for the "created_at" field
    in descending order.

    Args:
        subfeddit_id: The subfeddit's unique id.
        subfeddit_title: The subfeddit's title.
        enriched_comments: A list of comments, each with a polarity score
        and sentiment label.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        TypeError: If `subfeddit_id` is not an integer or `subfeddit_title`
        is not a string.
    """
    if not isinstance(subfeddit_id, int):
        raise TypeError("Subfeddit ID must be of type int")
    if not isinstance(subfeddit_title, str):
        raise TypeError("Subfeddit title must be of type string")

    return {
        "subfeddit": {
            "id": subfeddit_id,
            "title": subfeddit_title
        },
        "comment_count": len(enriched_comments),
        "sort": {
            "key": "created_at",
            "order": "desc"
        },
        "comments": enriched_comments
    }
