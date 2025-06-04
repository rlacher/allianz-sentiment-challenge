# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Integration tests with mocked external Feddit API requests."""
import pytest
from unittest.mock import patch, AsyncMock, Mock

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
    return {"subfeddits": [{"id": 1, "title": subfeddit_title}]}


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


def _make_mock_feddit_responses(valid_subfeddit: dict, valid_comments: dict):
    """Creates mock responses for subfeddit and comments API calls."""
    mock_subfeddit_response = Mock()
    mock_subfeddit_response.status_code = 200
    mock_subfeddit_response.json.return_value = valid_subfeddit
    mock_subfeddit_response.raise_for_status = Mock()

    mock_comments_response = Mock()
    mock_comments_response.status_code = 200
    mock_comments_response.json.return_value = valid_comments
    mock_comments_response.raise_for_status = Mock()

    return [mock_subfeddit_response, mock_comments_response]


@patch("feddit_sentiment.service.AsyncClient.get", new_callable=AsyncMock)
def test_valid_subfeddit_returns_sentiments(
    mock_get,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict,
    valid_comments: dict
):
    """Should return sentiment analysis for mocked comments."""
    mock_get.side_effect = _make_mock_feddit_responses(
        valid_subfeddit,
        valid_comments
    )

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={"subfeddit_title": subfeddit_title}
    )

    assert mock_get.await_count == 2
    assert response.status_code == 200

    data = response.json()
    assert data["subfeddit"]["title"] == subfeddit_title
    assert isinstance(data["comments"], list)
    assert data["sort"] == {"key": "created_at", "order": "desc"}
    assert data["comment_count"] == 3
    assert len(data["comments"]) == 3

    for comment in data["comments"]:
        assert comment["id"] in {101, 102, 103}
        assert "text" in comment
        assert isinstance(comment["polarity"], float)
        assert -1.0 <= comment["polarity"] <= 1.0
        assert comment["sentiment"] in {"positive", "negative"}

    for i in range(len(data["comments"]) - 1):
        assert data["comments"][i]["created_at"] >= \
            data["comments"][i + 1]["created_at"]


@patch("feddit_sentiment.service.AsyncClient.get", new_callable=AsyncMock)
def test_valid_subfeddit_sorted_by_polarity_asc(
    mock_get,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict,
    valid_comments: dict
):
    """Should return comments sorted by polarity in ascending order."""
    mock_get.side_effect = _make_mock_feddit_responses(
        valid_subfeddit,
        valid_comments
    )

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={
            "subfeddit_title": subfeddit_title,
            "polarity_sort_order": "asc"
        }
    )

    assert mock_get.await_count == 2
    assert response.status_code == 200

    data = response.json()
    comments = data["comments"]
    assert data["sort"] == {"key": "polarity", "order": "asc"}

    for i in range(len(comments) - 1):
        assert comments[i]["polarity"] <= comments[i + 1]["polarity"]


@patch("feddit_sentiment.service.AsyncClient.get", new_callable=AsyncMock)
def test_comments_filtered_by_time_range(
    mock_get,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict,
    valid_comments: dict
):
    """Should return only comments within the specified time range."""
    mock_get.side_effect = _make_mock_feddit_responses(
        valid_subfeddit,
        valid_comments
    )

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={
            "subfeddit_title": subfeddit_title,
            "time_from": 1748857610,
            "time_to": 1748857621
        }
    )

    assert mock_get.await_count == 2
    assert response.status_code == 200

    data = response.json()
    comments = data["comments"]
    assert data["comment_count"] == 2
    assert {c["id"] for c in comments} == {102, 103}

    for i in range(len(comments) - 1):
        assert comments[i]["created_at"] >= comments[i + 1]["created_at"]


@patch("feddit_sentiment.service.AsyncClient.get", new_callable=AsyncMock)
def test_subfeddit_with_no_comments_returns_empty_list(
    mock_get,
    client: TestClient,
    subfeddit_title: str,
    valid_subfeddit: dict
):
    """Should return empty comments list if subfeddit has no comments."""
    mock_get.side_effect = _make_mock_feddit_responses(
        valid_subfeddit,
        {"comments": []}
    )

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={"subfeddit_title": subfeddit_title}
    )

    assert mock_get.await_count == 2
    assert response.status_code == 200

    data = response.json()
    assert data["comments"] == []
    assert data["comment_count"] == 0


@patch("feddit_sentiment.service.AsyncClient.get", new_callable=AsyncMock)
def test_invalid_subfeddit_returns_404(
    mock_get,
    client: TestClient,
    valid_subfeddit: dict
):
    """Should return 404 for unknown subfeddit title."""
    mock_subfeddit_response = Mock()
    mock_subfeddit_response.status_code = 200
    mock_subfeddit_response.json.return_value = valid_subfeddit
    mock_subfeddit_response.raise_for_status = Mock()

    mock_get.return_value = mock_subfeddit_response

    response = client.get(
        f"/api/{API_VERSION}/comments",
        params={"subfeddit_title": "nonexistent"}
    )

    assert response.status_code == 404


def test_missing_subfeddit_title_returns_422(client: TestClient):
    """Should return 422 when required query param is missing."""
    response = client.get(f"/api/{API_VERSION}/comments")
    assert response.status_code == 422
