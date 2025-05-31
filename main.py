# SPDX-License-Identifier: MIT
# Copyright (c) 2025 René Lacher
"""Application entrypoint for Feddit Sentiment API"""
from fastapi import FastAPI
import uvicorn

from feddit_sentiment.core_service import router

# Initialise application
app = FastAPI(title="Feddit Sentiment API", version="1.0.0")

# Register router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
