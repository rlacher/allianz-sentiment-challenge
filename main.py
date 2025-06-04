# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Application entrypoint for Feddit Sentiment API"""
from fastapi import FastAPI
import uvicorn

from feddit_sentiment.routes import router
from feddit_sentiment.config import (
    APP_VERSION,
    API_HOST,
    API_PORT
)

# Initialise application
app = FastAPI(title="Feddit Sentiment API", version=APP_VERSION)

# Register router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT
    )
