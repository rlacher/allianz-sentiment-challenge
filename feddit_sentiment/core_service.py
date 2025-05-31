# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Core service of Feddit Sentiment API

Fetches comments from Feddit, applies sentiment classification, and returns
JSON output in an initial implementation.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/v1/comments")
async def get_comments_sentiment(subfeddit: str) -> dict:
    """Get subfeddit comments and analyze sentiment.

    Args:
        subfeddit: The subfeddit's name.
    Returns:
        JSON object with subfeddit name and dummy sentiment results.
    """
    return {
        "subfeddit": subfeddit,
        "comments": "Sentiment analysis results here"
    }
