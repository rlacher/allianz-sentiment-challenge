# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher

import pytest
from unittest.mock import (
    ANY,
    AsyncMock,
    patch
)

from feddit_sentiment.service import get_enriched_comments
from feddit_sentiment.schemas import SortOrder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subfeddit_title, polarity_sort, time_from, time_to, limit",
    [
        ("TechNews", None, None, None, 5),
        ("TechNews", SortOrder.asc, None, None, 5),
    ],
)
@patch("feddit_sentiment.service._fetch_subfeddits", new_callable=AsyncMock)
@patch("feddit_sentiment.service._find_subfeddit_id")
@patch(
    "feddit_sentiment.service._fetch_all_comments_lazy",
    new_callable=AsyncMock
)
@patch("feddit_sentiment.service._enrich_comments")
async def test_get_enriched_comments_calls_internal_functions(
    mock_enrich_comments,
    mock_fetch_comments,
    mock_find_subfeddit_id,
    mock_fetch_subfeddits,
    subfeddit_title,
    polarity_sort,
    time_from,
    time_to,
    limit,
):
    """Ensures get_enriched_comments correctly calls dependent functions."""
    mock_fetch_subfeddits.return_value = [{"title": subfeddit_title, "id": 1}]
    mock_find_subfeddit_id.return_value = 1
    mock_fetch_comments.return_value = []
    mock_enrich_comments.return_value = []

    await get_enriched_comments(
        subfeddit_title, polarity_sort, time_from, time_to, limit
    )

    mock_fetch_subfeddits.assert_awaited_once()
    mock_find_subfeddit_id.assert_called_once_with(
        mock_fetch_subfeddits.return_value, subfeddit_title
    )
    mock_fetch_comments.assert_awaited_once_with(1, ANY)
    mock_enrich_comments.assert_called_once_with(
        mock_fetch_comments.return_value
    )


@pytest.mark.asyncio
@patch("feddit_sentiment.service._fetch_subfeddits", new_callable=AsyncMock)
@patch("feddit_sentiment.service._find_subfeddit_id", return_value=1)
@pytest.mark.parametrize(
    "subfeddit_title, limit, exception_type, expected_message",
    [
        (None, 5, TypeError, "Subfeddit title must be of type str"),
        ("", 5, ValueError, "Subfeddit title must not be blank"),
        ("TechNews", -1, ValueError, "Limit must be a positive integer"),
    ],
)
async def test_get_enriched_comments_invalid_inputs(
    mock_fetch_subfeddits,
    mock_find_subfeddit_id,
    subfeddit_title,
    limit,
    exception_type,
    expected_message,
):
    """Ensures get_enriched_comments raises exceptions for invalid input."""
    with pytest.raises(exception_type, match=expected_message):
        await get_enriched_comments(
            subfeddit_title, None, 1748937600, 1748937600, limit
        )
