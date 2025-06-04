# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher

import pytest
from unittest.mock import patch

from fastapi import HTTPException

from feddit_sentiment.routes import get_comments_sentiment
from feddit_sentiment.schemas import CommentQueryParams


@pytest.fixture
def sample_enriched_comments():
    """Provides a sample list of sentiment-enriched comments."""
    return [
        {"id": 123, "text": "Great update!", "created_at": 1717395600,
         "polarity": 0.8, "sentiment": "positive"},
        {"id": 234, "text": "Terrible news.", "created_at": 1717392000,
         "polarity": -0.7, "sentiment": "negative"},
    ]


@pytest.mark.parametrize("subfeddit_title, subfeddit_id", [
    ("TechNews", 1),
])
@patch("feddit_sentiment.service.get_enriched_comments")
def test_get_comments_sentiment_happy_path(
    mock_get_enriched_comments,
    subfeddit_title,
    subfeddit_id,
    sample_enriched_comments
):
    """Ensures get_comments_sentiment returns correct formatted response."""
    mock_get_enriched_comments.return_value = (
        sample_enriched_comments, subfeddit_id
    )

    query = CommentQueryParams(subfeddit_title=subfeddit_title)
    response = get_comments_sentiment(query)

    assert response == {
        "subfeddit": {"id": subfeddit_id, "title": subfeddit_title},
        "comment_count": len(sample_enriched_comments),
        "filter": "None",
        "sort": {"key": "created_at", "order": "desc"},
        "comments": sample_enriched_comments
    }


@patch("feddit_sentiment.service.get_enriched_comments")
def test_get_comments_sentiment_with_polarity_sort(
    mock_get_enriched_comments,
    sample_enriched_comments
):
    """Checks sort metadata for polarity sort order."""
    subfeddit_title = "TechNews"
    subfeddit_id = 1
    sort_order = "asc"
    mock_get_enriched_comments.return_value = (
        sample_enriched_comments, subfeddit_id
    )

    query = CommentQueryParams(
        subfeddit_title=subfeddit_title,
        polarity_sort_order=sort_order
    )
    response = get_comments_sentiment(query)

    assert response["subfeddit"] == {
        "id": subfeddit_id,
        "title": subfeddit_title
    }
    assert response["comment_count"] == len(sample_enriched_comments)
    assert response["sort"] == {"key": "polarity", "order": sort_order}
    assert response["comments"] == sample_enriched_comments


@patch("feddit_sentiment.service.get_enriched_comments")
def test_get_comments_sentiment_with_time_range(
    mock_get_enriched_comments,
    sample_enriched_comments
):
    """Ensures response correctly reflects applied time range filtering."""
    subfeddit_title = "TechNews"
    subfeddit_id = 1
    time_from = 1717392000  # Excludes earlier comments
    time_to = 1717395600  # Includes comments up to this timestamp

    mock_get_enriched_comments.return_value = (
        sample_enriched_comments, subfeddit_id
    )

    query = CommentQueryParams(
        subfeddit_title=subfeddit_title,
        time_from=time_from,
        time_to=time_to
    )
    response = get_comments_sentiment(query)

    assert response["subfeddit"] == {
        "id": subfeddit_id,
        "title": subfeddit_title
    }
    assert response["comment_count"] == len(sample_enriched_comments)
    assert response["filter"] == {
        "time_from": time_from,
        "time_to": time_to
    }, "Expected filter metadata to match time range params"
    assert response["comments"] == sample_enriched_comments


@pytest.mark.parametrize("exception_message", [
    "Subfeddit 'TechNews' not found.",
    "Invalid JSON response for subfeddits",
])
@patch("feddit_sentiment.service.get_enriched_comments")
def test_get_comments_sentiment_subfeddit_not_found(
    mock_get_enriched_comments, exception_message
):
    """Ensures get_comments_sentiment raises a 404 HTTPException."""
    mock_get_enriched_comments.side_effect = ValueError(exception_message)

    with pytest.raises(HTTPException) as exc_info:
        query = CommentQueryParams(subfeddit_title="TechNews")
        get_comments_sentiment(query)

    assert exc_info.value.status_code == 404
    assert exception_message in str(exc_info.value.detail)
