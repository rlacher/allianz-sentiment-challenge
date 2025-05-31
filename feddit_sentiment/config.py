# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Configuration settings for the Feddit Sentiment API"""
from os import getenv

# Versioning
API_VERSION = "v1"
APP_VERSION = "1.0.0"

# Default settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Environment overrides
API_HOST = getenv("API_HOST", DEFAULT_HOST)
API_PORT = int(getenv("API_PORT", DEFAULT_PORT))
RELOAD_MODE = True
