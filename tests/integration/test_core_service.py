# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
import pytest
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

from main import app
from feddit_sentiment.config import API_VERSION


class TestGetCommentsSentiment:
    """Integration tests with mocked external Feddit API requests."""

    SUBFEDDIT_TITLE = "tech"

    @pytest.fixture
    def client(self) -> TestClient:
        """Initialises and returns the test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_subfeddit(self) -> dict:
        """Returns a valid subfeddit as dictionary."""
        return {"subfeddits": [
            {"id": 1, "title": self.SUBFEDDIT_TITLE}
        ]}

    @pytest.fixture
    def valid_comments(self) -> dict:
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
                }
            ]
        }

    @patch("feddit_sentiment.core_service.requests.get")
    def test_valid_subfeddit_returns_sentiments(
        self,
        mock_get: Mock,
        client: TestClient,
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
                "subfeddit_title": self.SUBFEDDIT_TITLE
            }
        )

        assert response.status_code == 200, "Expected 200 OK from endpoint"
        data = response.json()

        assert data["title"] == self.SUBFEDDIT_TITLE, \
            "Expected correct title"
        assert isinstance(data["comments"], list), "Expected comments as list"
        comments = data["comments"]

        assert len(comments) == 2, \
            "Expected 2 mocked comments returned"

        for comment in comments:
            assert comment["id"] in {101, 102}, "Unexpected comment ID"
            assert "text" in comment, "Comment missing 'text' field"
            assert "polarity" in comment, "Expected polarity score field"
            assert isinstance(comment["polarity"], float), \
                "Expected polarity score to be a float"
            assert -1.0 <= comment["polarity"] <= 1.0, \
                "Expected polarity score within valid range"
            assert comment["sentiment"] in {"positive", "negative"}, \
                "Invalid sentiment value"

        for i in range(len(comments)-1):
            curr_created_at = comments[i]["created_at"]
            next_created_at = comments[i+1]["created_at"]
            assert curr_created_at >= next_created_at, \
                f"Comments not sorted in descending order at index {i}"

    @patch("feddit_sentiment.core_service.requests.get")
    def test_subfeddit_with_no_comments_returns_empty_list(
        self,
        mock_get: Mock,
        client: TestClient,
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
            params={"subfeddit_title": self.SUBFEDDIT_TITLE}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["title"] == self.SUBFEDDIT_TITLE
        assert isinstance(data["comments"], list)
        assert data["comments"] == [], "Expected empty comments list"

    @patch("feddit_sentiment.core_service.requests.get")
    def test_invalid_subfeddit_returns_404(
        self,
        mock_get: Mock,
        client: TestClient,
    ):
        """Should return 404 for unknown subfeddit title."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "subfeddits": [{"id": 1, "title": "tech"}]
        }

        mock_get.return_value = mock_response

        response = client.get(
            f"/api/{API_VERSION}/comments",
            params={"subfeddit_title": "nonexistent"}
        )

        assert response.status_code == 404, "Expected 404 Not Found"

    def test_missing_subfeddit_title_returns_422(
        self,
        client: TestClient
    ):
        """Should return 422 when required query param is missing."""
        response = client.get(f"/api/{API_VERSION}/comments")
        assert response.status_code == 422, \
            "Expected 422 Unprocessable Entity for missing param"
