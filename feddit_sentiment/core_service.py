# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Core service of Feddit Sentiment API

Fetches comments from Feddit, applies sentiment classification, and returns
JSON output in an initial implementation.
"""
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

MAX_COMMENTS_TO_FETCH = 25

router = APIRouter()


@router.get(f"/api/{API_VERSION}/comments")
def get_comments_sentiment(subfeddit_title: str) -> dict:
    """Get subfeddit comments and analyse sentiment.

    Args:
        subfeddit_title: The subfeddit's title.
    Returns:
        Dictionary containing the subfeddit title, comment limit,
        and a list of comments with sentiment scores.
    Raises:
        HTTPException: If the subfeddit does not exist.
    """
    subfeddits = _fetch_subfeddits()

    subfeddit_id = next(
        (s['id'] for s in subfeddits if s['title'] == subfeddit_title),
        None
    )

    if subfeddit_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Subfeddit {subfeddit_title} not found"
        )

    comments = _fetch_comments(subfeddit_id)

    analyser = SentimentIntensityAnalyzer()
    sentiment_results = []
    for comment in comments:
        comment_text = comment.get('text')
        polarity_score = _analyse_comment(analyser, comment_text)

        sentiment_results.append({
            "id": comment.get('id'),
            "text": comment_text,
            "polarity": polarity_score,
            "sentiment": "positive" if polarity_score > 0 else "negative"
        })

    return {
        "title": subfeddit_title,
        "limit": MAX_COMMENTS_TO_FETCH,
        "comments": sentiment_results
    }


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
        return subfeddits
    except RequestException as request_exception:
        raise ValueError("Failed to get subfeddits") from request_exception
    except JSONDecodeError as json_error:
        raise ValueError("Invalid JSON response") from json_error


def _fetch_comments(subfeddit_id: int) -> list:
    """Retieves comments from the subfeddit by id.

    Args:
        subfeddit_id: The subfeddit's id.
    Returns:
        A list of comments for the subfeddit.
    Raises:
        TypeError: If subfeddit_id is not an integer.
        ValueError: If the HTTP request fails or if the API response body
        is not valid JSON.
    """
    # Fetch oldest comments for now
    if not isinstance(subfeddit_id, int):
        raise TypeError("Subfeddit ID must be of type int")

    url = f"http://{FEDDIT_HOST}:{FEDDIT_PORT}/api/v1/comments"
    params = {"subfeddit_id": subfeddit_id, "limit": MAX_COMMENTS_TO_FETCH}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        comments = data.get('comments', [])
        return comments
    except RequestException as request_exception:
        raise ValueError("Failed to get comments") from request_exception
    except JSONDecodeError as json_error:
        raise ValueError("Invalid JSON response") from json_error


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
    return compound_score
