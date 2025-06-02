# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Core service of Feddit Sentiment API

Fetches comments from Feddit, applies sentiment classification, and returns
JSON output in an initial implementation.
"""
import logging
import requests
from requests.exceptions import RequestException
from json import JSONDecodeError

from fastapi import APIRouter, HTTPException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from feddit_sentiment.config import (
    API_VERSION,
    FEDDIT_HOST,
    FEDDIT_PORT
)

COMMENTS_PER_REQUEST = 500
NUM_RECENT_COMMENTS = 25
MAX_COMMENT_PRINT_LENGTH = 30

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(f"/api/{API_VERSION}/comments")
def get_comments_sentiment(subfeddit_title: str) -> dict:
    """Get subfeddit comments and analyse sentiment.

    Args:
        subfeddit_title: The subfeddit's title.
    Returns:
        Dictionary containing the subfeddit title, comment limit
        and a list of the most recent comments with sentiment scores.
    Raises:
        HTTPException: If the subfeddit does not exist.
    """
    subfeddits = _fetch_subfeddits()

    subfeddit_id = next(
        (s['id'] for s in subfeddits if s['title'] == subfeddit_title),
        None
    )

    if subfeddit_id is None:
        logger.warning(f"Subfeddit '{subfeddit_title}' not found.")
        raise HTTPException(
            status_code=404,
            detail=f"Subfeddit '{subfeddit_title}' not found"
        )

    comments = _fetch_all_comments_exhaustively(subfeddit_id)

    # Sort comments in-place by creation timestamp descending
    comments.sort(key=lambda comment: comment["created_at"], reverse=True)

    recent_comments = comments[:NUM_RECENT_COMMENTS]

    analyser = SentimentIntensityAnalyzer()
    sentiment_results = []
    for comment in recent_comments:
        comment_text = comment.get('text')
        polarity_score = _analyse_comment(analyser, comment_text)

        sentiment_results.append({
            "id": comment.get('id'),
            "text": comment_text,
            "created_at": comment.get('created_at'),
            "polarity": polarity_score,
            "sentiment": "positive" if polarity_score > 0 else "negative"
        })

    logger.info(
        f"Completed sentiment analysis for {len(sentiment_results)} comments "
        f"from subfeddit '{subfeddit_title}'."
    )

    return _create_output_structure(
        subfeddit_id,
        subfeddit_title,
        sentiment_results
    )


def _fetch_subfeddits() -> list:
    """Retrieve the list of available subfeddits.

    Returns:
        A list of subfeddits.
    Raises:
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    # Assumes all subfeddits are returned at once
    url = f"http://{FEDDIT_HOST}:{FEDDIT_PORT}/api/v1/subfeddits"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        subfeddits = data.get('subfeddits', [])

        logger.info(f"Fetched {len(subfeddits)} subfeddits")

        return subfeddits
    except RequestException as request_exception:
        logger.warning("Failed to get subfeddits", exc_info=True)
        raise ValueError("Failed to get subfeddits") from request_exception
    except JSONDecodeError as json_error:
        logger.warning("Invalid JSON response for subfeddits", exc_info=True)
        raise ValueError(
            "Invalid JSON response for subfeddits"
        ) from json_error


def _fetch_all_comments_exhaustively(subfeddit_id: int) -> list:
    """Exhaustively fetches all comments from subfeddit by id.

    Paginates to retrieve all comments from the Feddit API that only
    supports 'limit' and 'skip' query parameters.

    Args:
        subfeddit_id: The subfeddit's id.
    Returns:
        A list of all comments for the subfeddit.
    Raises:
        TypeError: If subfeddit_id is not an integer.
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    if not isinstance(subfeddit_id, int):
        raise TypeError("Subfeddit ID must be of type int")

    # Built-in list has O(1) resize/append complexity
    all_comments = []
    skip_offset = 0

    url = f"http://{FEDDIT_HOST}:{FEDDIT_PORT}/api/v1/comments"
    params = {
        "subfeddit_id": subfeddit_id,
        "limit": COMMENTS_PER_REQUEST
    }

    while True:
        try:
            params["skip"] = skip_offset
            response = requests.get(url, params)

            response.raise_for_status()
            data = response.json()
            comments = data.get('comments', [])
            num_comments = len(comments)

            logger.debug(
                f"Fetched {num_comments} comments "
                f"for subfeddit with id {subfeddit_id}, "
                f"skipping {skip_offset}"
            )

            all_comments.extend(comments)

            if num_comments < COMMENTS_PER_REQUEST:
                logger.info(
                    f"Fetched all {len(all_comments)} comments "
                    f"for subfeddit with id {subfeddit_id}"
                )
                break
            else:
                skip_offset += COMMENTS_PER_REQUEST

        except RequestException as request_exception:
            logger.warning("Failed to get comments", exc_info=True)
            raise ValueError("Failed to get comments") from request_exception
        except JSONDecodeError as json_error:
            logger.warning(
                "Invalid JSON response for comments",
                exc_info=True
            )
            raise ValueError(
                "Invalid JSON response for comments"
            ) from json_error

    return all_comments


def _analyse_comment(
        analyser: SentimentIntensityAnalyzer,
        comment_text: str) -> float:
    """Analyses the sentiment of a comment.

    Args:
        analyser: The sentiment analyser instance.
        comment_text: The comment's raw text to analyse.
    Returns:
        A polarity score from -1 to 1.
    Raises:
        TypeError: If comment_text is not a string.
        ValueError: If comment_text is a blank string.
    """
    if not isinstance(comment_text, str):
        raise TypeError("Comment must be of type str")
    if not comment_text.strip():
        raise ValueError("Comment must not be blank")

    scores = analyser.polarity_scores(comment_text)
    compound_score = scores.get('compound')

    logger.debug(
        f"Polarity score {compound_score:.2f} "
        f"for text: {comment_text[:MAX_COMMENT_PRINT_LENGTH]}"
        f"{"..." if len(comment_text) >= MAX_COMMENT_PRINT_LENGTH else ""}"
    )

    return compound_score


def _create_output_structure(
        subfeddit_id: int,
        subfeddit_title: str,
        sentiment_results: list
) -> dict:
    """Maps meta and comment data into output dictionary.

    Information regarding sorting is hardcoded for the "created_at" field
    in descending order.

    Args:
        subfeddit_id: The subfeddit's unique id.
        subfeddit_title: The subfeddit's title.
        sentiment_results: A list of comments, each with a polarity score
        and sentiment label.
    Returns:
        A structured dictionary containing subfeddit metadata, sorting info,
        and sentiment-labelled comments.
    Raises:
        TypeError: If 'subfeddit_id' is not an integer or 'subfeddit_title'
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
        "comment_count": len(sentiment_results),
        "sort": {
            "key": "created_at",
            "order": "desc"
        },
        "comments": sentiment_results
    }
