# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Feddit Sentiment API

Initialises basic logging configuration.
"""
import logging

from feddit_sentiment import config


def configure_logging() -> None:
    """Set up basic console logging."""
    logging.basicConfig(
        level=config.LOGGING_LEVEL,
        format=config.LOGGING_FORMAT
    )


configure_logging()
