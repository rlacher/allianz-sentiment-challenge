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
        """Initialises and retuns the test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_subfeddit(self) -> dict:
        """Returns a valid subfeddit as dictionary."""
        return {"subfeddits": [
            {"id": 1, "title": TestGetCommentsSentiment.SUBFEDDIT_TITLE}
        ]}

    @pytest.fixture
    def valid_comments(self) -> dict:
        """Returns a valid dictionary of comments."""
        return {
            "comments": [
                {"id": 101, "text": "I love this post!"},
                {"id": 102, "text": "Terrible idea, not impressed."}
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

        # Patch sequential calls to requests.get
        mock_get.side_effect = [
            mock_subfeddits_response,
            mock_comments_response
        ]

        response = client.get(
            f"/api/{API_VERSION}/comments",
            params={
                "subfeddit_title": TestGetCommentsSentiment.SUBFEDDIT_TITLE
            }
        )

        assert response.status_code == 200, "Expected 200 OK from endpoint"
        data = response.json()

        assert data["title"] == TestGetCommentsSentiment.SUBFEDDIT_TITLE, \
            "Expected correct title"
        assert isinstance(data["comments"], list), "Expected comments as list"
        assert len(data["comments"]) == 2, \
            "Expected 2 mocked comments returned"

        for comment in data["comments"]:
            assert "polarity" in comment, "Expected polarity score field"
            assert isinstance(comment["polarity"], float), \
                "Expected polarity score to be a float"
            assert -1.0 <= comment["polarity"] <= 1.0, \
                "Expected polarity score within valid range"
            assert comment["sentiment"] in {"positive", "negative"}, \
                "Invalid sentiment value"

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
