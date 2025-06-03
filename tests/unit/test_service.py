# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher

import pytest
from unittest.mock import patch

from feddit_sentiment.service import get_enriched_comments


@pytest.fixture
def sample_comments():
    """Provides a sample list of comments for testing."""
    return [
        {"id": 1, "text": "Great update!", "created_at": "1748937580"},
        {"id": 2, "text": "Terrible news.", "created_at": "1748937590"},
    ]


@pytest.mark.parametrize("subfeddit_title, limit", [
    ("TechNews", 5),
    ("Movies", 10),
])
@patch("feddit_sentiment.service._fetch_subfeddits",
       return_value=[{"title": "TechNews", "id": 1}])
@patch("feddit_sentiment.service._find_subfeddit_id", return_value=1)
@patch("feddit_sentiment.service._fetch_all_comments_exhaustively")
@patch("feddit_sentiment.service._enrich_comments")
def test_get_enriched_comments_happy_path(
    mock_enrich_comments,
    mock_fetch_comments,
    mock_find_subfeddit_id,
    mock_fetch_subfeddits,
    subfeddit_title,
    limit,
    sample_comments
):
    """Ensures get_enriched_comments functions correctly on the happy path."""
    mock_fetch_comments.return_value = sample_comments
    mock_enrich_comments.return_value = [
        {**comment, "polarity": 0.8 if comment["id"] == 1 else -0.7,
         "sentiment": "positive" if comment["id"] == 1 else "negative"}
        for comment in sample_comments
    ]

    comments, subfeddit_id = get_enriched_comments(subfeddit_title, limit)

    assert subfeddit_id == 1
    assert len(comments) <= limit
    assert all(
        "polarity" in comment and "sentiment" in comment
        for comment in comments
    )


@pytest.mark.parametrize("subfeddit_title, exception_type, expected_message", [
    (None, TypeError, "Subfeddit title must be of type str"),
    ("", ValueError, "Subfeddit title must not be blank"),
    ("TechNews", ValueError, "Limit must be a positive integer"),
])
def test_get_enriched_comments_invalid_inputs(
        subfeddit_title, exception_type, expected_message):
    """Ensures get_enriched_comments raises exceptions for invalid input."""
    with pytest.raises(exception_type, match=expected_message):
        get_enriched_comments(subfeddit_title, -1)
