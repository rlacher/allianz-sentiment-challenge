# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Service layer of Feddit Sentiment API

Encompasses the API's business logic: Fetch comments from Feddit,
comment processing and sentiment classification.
"""
from httpx import (
    AsyncClient,
    RequestError,
    HTTPStatusError
)
from json import JSONDecodeError
import logging
from sys import maxsize as MAXSIZE

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from feddit_sentiment.config import (
    FEDDIT_HOST,
    FEDDIT_PORT
)
from feddit_sentiment.schemas import SortOrder

COMMENTS_PER_REQUEST = 500
FEDDIT_BASE_PATH = f"http://{FEDDIT_HOST}:{FEDDIT_PORT}/api/v1"
MAX_COMMENT_PRINT_LENGTH = 30

logger = logging.getLogger(__name__)


async def get_enriched_comments(
        subfeddit_title: str,
        polarity_sort: SortOrder | None,
        time_from: int | None,
        time_to: int | None,
        limit: int,
) -> tuple[list[dict], int]:
    """Fetch, optionally filter/sort, and enrich comments from a subfeddit.

    Args:
        subfeddit_title: The subfeddit's title.
        polarity_sort: Polarity sort order applied to the
        most recent comments or None for no sorting.
        time_from: UNIX timestamp to filter comments from (inclusive).
        time_to: UNIX timestamp to filter comments before (exclusive).
        limit: The maximum number of comments to return.
    Returns:
        A list of enriched comments with sentiment polarity and label.
    Raises:
        TypeError: If `subfeddit_title` is not a string.
        ValueError: If `subfeddit_title` is blank or if `limit` is not
        a positive integer.
    """
    if not isinstance(subfeddit_title, str):
        raise TypeError("Subfeddit title must be of type str")
    if not subfeddit_title.strip():
        raise ValueError("Subfeddit title must not be blank")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer")

    async with AsyncClient() as client:
        subfeddits = await _fetch_subfeddits(client)
        subfeddit_id = _find_subfeddit_id(subfeddits, subfeddit_title)
        raw_comments = await _fetch_all_comments_lazy(subfeddit_id, client)

    # Order comments by most recent first
    raw_comments.sort(key=lambda c: c["created_at"], reverse=True)
    # Filter comments by time range
    raw_comments = _filter_comments_by_time(
        raw_comments,
        time_from,
        time_to
    )

    enriched_comments = _enrich_comments(raw_comments[:limit])

    if polarity_sort is not None:
        enriched_comments = _sort_comments_by_polarity(
            enriched_comments,
            polarity_sort
        )

    return enriched_comments, subfeddit_id


async def _fetch_subfeddits(client: AsyncClient) -> list:
    """Retrieve the list of available subfeddits.

    Args:
        client: The asynchronous HTTP client to use.
    Returns:
        A list of subfeddits.
    Raises:
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    # Assumes all subfeddits are returned at once
    url = f"{FEDDIT_BASE_PATH}/subfeddits/"
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        subfeddits = data.get('subfeddits', [])

        logger.info(f"Fetched {len(subfeddits)} subfeddits")

        return subfeddits
    except (RequestError, HTTPStatusError) as request_exception:
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


async def _fetch_all_comments_lazy(
        subfeddit_id: int,
        client: AsyncClient
) -> list:
    """Fetch all comments for a given subfeddit using lazy pagination.

    Paginates to retrieve all comments from the Feddit API that only
    supports `limit` and `skip` query parameters.

    Args:
        subfeddit_id: The subfeddit's id.
        client: The asynchronous HTTP client to use.
    Returns:
        A list of comment dictionaries for the subfeddit.
    Raises:
        TypeError: If `subfeddit_id` is not an integer.
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    if not isinstance(subfeddit_id, int):
        raise TypeError("Subfeddit ID must be of type int")

    url = f"{FEDDIT_BASE_PATH}/comments/"

    # Built-in list has O(1) resize/append complexity
    all_comments = []
    skip_offset = 0

    while True:
        params = {
            "subfeddit_id": subfeddit_id,
            "limit": COMMENTS_PER_REQUEST,
            "skip": skip_offset
        }
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            comment_page = data.get('comments', [])
            num_comments = len(comment_page)

            logger.debug(
                f"Fetched {num_comments} comments "
                f"for subfeddit with id {subfeddit_id}, "
                f"skipping {skip_offset}"
            )

        except (RequestError, HTTPStatusError) as request_error:
            logger.warning(
                f"Failed to fetch comments at skip: {skip_offset}",
                exc_info=True
            )
            raise ValueError("Failed to fetch comments") from request_error
        except JSONDecodeError as json_error:
            logger.warning("Invalid JSON for comment page", exc_info=True)
            raise ValueError("Invalid JSON for comment page") from json_error

        all_comments.extend(comment_page)

        if not comment_page or len(comment_page) < COMMENTS_PER_REQUEST:
            logger.info(
                f"Fetched all {len(all_comments)} comments "
                f"for subfeddit with id {subfeddit_id}"
            )
            break

        skip_offset += COMMENTS_PER_REQUEST

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

    results = []

    for i, comment in enumerate(comments):
        try:
            comment_id = comment['id']
            text = comment['text']
            created_at = comment['created_at']

            polarity_score = _analyse_comment(text)

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


def _get_analyser() -> SentimentIntensityAnalyzer:
    """Provides a reusable SentimentIntensityAnalyzer instance.

    Returns:
        A SentimentIntensityAnalyzer instance.
    """
    return SentimentIntensityAnalyzer()


def _analyse_comment(
        comment_text: str,
        analyser: SentimentIntensityAnalyzer = _get_analyser()
) -> float:
    """Analyses the sentiment of a comment.

    Args:
        comment_text: The comment's raw text to analyse.
        analyser: The sentiment analyser instance.
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


def _sort_comments_by_polarity(
    comments: list[dict],
    order: SortOrder
) -> list[dict]:
    """Sorts comments by polarity score.

    Args:
        comments: List of enriched comments.
        order: SortOrder enum (asc or desc).
    Returns:
        A new list sorted by polarity.
    """
    reverse = (order == SortOrder.desc)
    sorted_comments = sorted(
        comments,
        key=lambda c: c["polarity"],
        reverse=reverse
    )

    logger.info(
        f"Sorted {len(comments)} comments by polarity in "
        f"{'descending' if reverse else 'ascending'} order"
    )

    return sorted_comments


def _filter_comments_by_time(
    comments: list[dict],
    time_from: int | None,
    time_to: int | None
) -> list[dict]:
    """Filters comments by a time range.

    Args:
        comments: List of enriched comments.
        time_from: UNIX timestamp to filter from or None for no lower bound.
        time_to: UNIX timestamp to filter up to or None for no upper bound.
    Returns:
        A new list of comments filtered by the specified time range.
    """
    if time_from is None and time_to is None:
        return comments
    elif time_from is None:
        time_from = 0
    elif time_to is None:
        time_to = MAXSIZE

    filtered_comments = [
        c for c in comments
        if c["created_at"] >= time_from and c["created_at"] < time_to
    ]

    logger.info(
        f"Filtered {len(comments)} comments to {len(filtered_comments)} "
        f"by time range ({time_from}, {time_to})"
    )

    return filtered_comments
