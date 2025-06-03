# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Service layer of Feddit Sentiment API

Encompasses the API's business logic: Fetch comments from Feddit,
comment processing and sentiment classification.
"""
import logging
import requests
from requests.exceptions import RequestException
from json import JSONDecodeError

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from feddit_sentiment.config import (
    FEDDIT_HOST,
    FEDDIT_PORT
)

COMMENTS_PER_REQUEST = 500
FEDDIT_BASE_PATH = f"http://{FEDDIT_HOST}:{FEDDIT_PORT}/api/v1"
MAX_COMMENT_PRINT_LENGTH = 30

logger = logging.getLogger(__name__)


def get_enriched_comments(
        subfeddit_title: str,
        limit: int
) -> tuple[list[dict], int]:
    """Fetches and processes comments from a subfeddit.

    Args:
        subfeddit_title: The subfeddit's title.
        limit: The maximum number of comments to return.
    Returns:
        A list of enriched comments with sentiment polarity and label.
    Raises:
        TypeError: If the subfeddit title is not a string.
        ValueError: If the subfeddit title is blank or if the limit is
        not a positive integer.
    """
    if not isinstance(subfeddit_title, str):
        raise TypeError("Subfeddit title must be of type str")
    if not subfeddit_title.strip():
        raise ValueError("Subfeddit title must not be blank")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer")

    subfeddits = _fetch_subfeddits()
    subfeddit_id = _find_subfeddit_id(subfeddits, subfeddit_title)
    raw_comments = _fetch_all_comments_exhaustively(subfeddit_id)

    # Order comments by most recent first
    raw_comments.sort(key=lambda comment: comment["created_at"], reverse=True)

    return _enrich_comments(raw_comments[:limit]), subfeddit_id


def _fetch_subfeddits() -> list:
    """Retrieve the list of available subfeddits.

    Returns:
        A list of subfeddits.
    Raises:
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    # Assumes all subfeddits are returned at once
    url = f"{FEDDIT_BASE_PATH}/subfeddits"
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


def _find_subfeddit_id(
        subfeddits: list[dict],
        subfeddit_title: str
) -> int:
    """Finds the subfeddit's id by its title case-insensitively.

    Args:
        subfeddits: A list of subfeddit dictionaries.
        subfeddit_title: The subfeddit's title.
    Returns:
        The subfeddit's unique id.
    Raises:
        TypeError: If `subfeddits` is not a list or `subfeddit_title` is not
        a string.
        ValueError: If the subfeddit with the given title does not exist.
    """
    if not isinstance(subfeddits, list):
        raise TypeError("Subfeddits must be of list type")
    if not isinstance(subfeddit_title, str):
        raise TypeError("Subfeddit title must be of type str")

    for subfeddit in subfeddits:
        if subfeddit['title'].lower() == subfeddit_title.lower():
            return subfeddit['id']
    raise ValueError(f"Subfeddit '{subfeddit_title}' not found.")


def _fetch_all_comments_exhaustively(subfeddit_id: int) -> list:
    """Exhaustively fetches all comments from subfeddit by id.

    Paginates to retrieve all comments from the Feddit API that only
    supports `limit` and `skip` query parameters.

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

    url = f"{FEDDIT_BASE_PATH}/comments"
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


def _enrich_comments(comments: list) -> list:
    """Enriches comments with polarity score and sentiment label.

    Iterates over a list of comment dictionaries and enriches each with a
    sentiment polarity and label by calling `_analyse_comment()`.

    Args:
        comments: A list of comment dictionaries, each containing meta data.
    Returns:
        A list of enriched comments with added `polarity` and `sentiment`
        fields.
    Raises:
        TypeError: If `comments` is not a list.
        ValueError: If any comment is missing a mandatory field.
    """
    if not isinstance(comments, list):
        raise TypeError("Comments must be of list type")

    analyser = SentimentIntensityAnalyzer()
    results = []

    for i, comment in enumerate(comments):
        try:
            comment_id = comment['id']
            text = comment['text']
            created_at = comment['created_at']

            polarity_score = _analyse_comment(analyser, text)

            results.append({
                "id": comment_id,
                "text": text,
                "created_at": created_at,
                "polarity": polarity_score,
                "sentiment": "positive" if polarity_score > 0 else "negative"
            })
        except KeyError as key_error:
            error_message = (
                f"Comment at index {i} is missing a mandatory field: "
                f"{key_error}."
            )
            logger.warning(error_message)
            raise ValueError(error_message) from key_error

    logger.info(
        f"Completed sentiment analysis for {len(results)} comments."
    )

    return results


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
        TypeError: If `comment_text` is not a string.
        ValueError: If `comment_text` is a blank string.
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
