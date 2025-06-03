# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Integration tests with mocked external Feddit API requests."""
import pytest
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

from main import app
from feddit_sentiment.config import API_VERSION


@pytest.fixture
def subfeddit_title() -> str:
    """Fixture to provide a subfeddit title for tests."""
    return "TechNews"


@pytest.fixture
def client() -> TestClient:
    """Initialises and returns the test client."""
    return TestClient(app)


@pytest.fixture
def valid_subfeddit(subfeddit_title: str) -> dict:
    """Returns a valid subfeddit as dictionary."""
    return {"subfeddits": [
        {"id": 1, "title": subfeddit_title}
    ]}


@pytest.fixture
def valid_comments() -> dict:
    """Returns a valid dictionary of comments."""
    return {
        "comments": [
            {
                "id": 101,
                "text": "I love this post!",
                "created_at": 1748857600
            },
            {
                "id": 102,
                "text": "Terrible idea, not impressed.",
                "created_at": 1748857610
            },
            {
                "id": 103,
                "text": "It's not all that bad.",
                "created_at": 1748857620
            }
        ]
    }


@patch("feddit_sentiment.service.requests.get")
def test_valid_subfeddit_returns_sentiments(
    mock_get: Mock,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict,
    valid_comments: dict
):
    """Should return sentiment analysis for mocked comments."""
    # Mock /subfeddits response
    mock_subfeddits_response = Mock()
    mock_subfeddits_response.raise_for_status.return_value = None
    mock_subfeddits_response.json.return_value = valid_subfeddit

    # Mock /comments response
    mock_comments_response = Mock()
    mock_comments_response.raise_for_status.return_value = None
    mock_comments_response.json.return_value = valid_comments

    # Simulate sequential calls
    mock_get.side_effect = [
        mock_subfeddits_response,
        mock_comments_response
    ]

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={
            "subfeddit_title": subfeddit_title
        }
    )

    assert mock_get.call_count == 2, \
        "Expected two calls: one for subfeddits, one for comments"

    assert response.status_code == 200, "Expected 200 OK from endpoint"
    data = response.json()

    assert data["subfeddit"]["title"] == subfeddit_title, \
        "Expected correct subfeddit title"
    assert isinstance(data["comments"], list), "Expected comments as list"

    assert data["sort"] == {
        "key": "created_at",
        "order": "desc"
    }, "Expected correct hardcoded sort metadata"

    comments = data["comments"]
    assert data["comment_count"] == 3, \
        "Expected comment count to match number of input comments"
    assert len(comments) == data["comment_count"], \
        "Expected comment count to match length of comments list"

    for comment in comments:
        assert comment["id"] in {101, 102, 103}, "Unexpected comment ID"
        assert "text" in comment, "Comment missing 'text' field"
        assert "polarity" in comment, "Expected polarity score field"
        assert isinstance(comment["polarity"], float), \
            "Expected polarity score to be a float"
        assert -1.0 <= comment["polarity"] <= 1.0, \
            "Expected polarity score within valid range"
        assert comment["sentiment"] in {"positive", "negative"}, \
            "Invalid sentiment value"

    for i in range(len(comments) - 1):
        curr_created_at = comments[i]["created_at"]
        next_created_at = comments[i + 1]["created_at"]
        assert curr_created_at >= next_created_at, \
            f"Comments not sorted in descending order at index {i}"


@patch("feddit_sentiment.service.requests.get")
def test_valid_subfeddit_sorted_by_polarity_asc(
    mock_get: Mock,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict,
    valid_comments: dict
):
    """Should return comments sorted by polarity in ascending order."""
    # Mock /subfeddits response
    mock_subfeddits_response = Mock()
    mock_subfeddits_response.raise_for_status.return_value = None
    mock_subfeddits_response.json.return_value = valid_subfeddit

    # Mock /comments response
    mock_comments_response = Mock()
    mock_comments_response.raise_for_status.return_value = None
    mock_comments_response.json.return_value = valid_comments

    mock_get.side_effect = [
        mock_subfeddits_response,
        mock_comments_response
    ]

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={
            "subfeddit_title": subfeddit_title,
            "polarity_sort_order": "asc"
        }
    )

    assert mock_get.call_count == 2, \
        "Expected two calls: one for subfeddits, one for comments"

    assert response.status_code == 200
    data = response.json()
    comments = data["comments"]

    assert data["comment_count"] == 3, \
        "Expected comment count to match number of input comments"
    assert len(comments) == data["comment_count"], \
        "Expected comment count to match length of comments list"
    assert data["sort"] == {
        "key": "polarity",
        "order": "asc"
    }, "Expected sort metadata to reflect polarity sort"

    for i in range(len(comments) - 1):
        curr_polarity = comments[i]["polarity"]
        next_polarity = comments[i + 1]["polarity"]
        assert curr_polarity <= next_polarity, \
            f"Comments not sorted by ascending polarity at index {i}"


@patch("feddit_sentiment.service.requests.get")
def test_subfeddit_with_no_comments_returns_empty_list(
    mock_get: Mock,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict
):
    """Should return empty comments list if subfeddit has no comments."""

    # Mock /subfeddits response
    mock_subfeddits_response = Mock()
    mock_subfeddits_response.raise_for_status.return_value = None
    mock_subfeddits_response.json.return_value = valid_subfeddit

    # Mock /comments response (empty)
    mock_comments_response = Mock()
    mock_comments_response.raise_for_status.return_value = None
    mock_comments_response.json.return_value = {"comments": []}

    # Simulate sequential calls
    mock_get.side_effect = [
        mock_subfeddits_response,
        mock_comments_response
    ]

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={"subfeddit_title": subfeddit_title}
    )

    assert mock_get.call_count == 2, \
        "Expected two calls: one for subfeddits, one for comments"

    assert response.status_code == 200
    data = response.json()

    assert data["subfeddit"]["title"] == subfeddit_title
    assert isinstance(data["comments"], list)
    assert data["comments"] == [], "Expected empty comments list"
    assert data["comment_count"] == 0, "Expected comment count of 0"


@patch("feddit_sentiment.service.requests.get")
def test_invalid_subfeddit_returns_404(
    mock_get: Mock,
    client: TestClient,
):
    """Should return 404 for unknown subfeddit title."""
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "subfeddits": [{"id": 1, "title": "TechNews"}]
    }

    mock_get.return_value = mock_response

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={"subfeddit_title": "nonexistent"}
    )

    assert response.status_code == 404, "Expected 404 Not Found"


def test_missing_subfeddit_title_returns_422(
    client: TestClient
):
    """Should return 422 when required query param is missing."""
    response = client.get(f"/api/{API_VERSION}/comments")
    assert response.status_code == 422, \
        "Expected 422 Unprocessable Entity for missing param"
