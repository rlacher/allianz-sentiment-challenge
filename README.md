# Allianz Sentiment Challenge

<!-- Badges -->
[![flake8](https://img.shields.io/github/actions/workflow/status/rlacher/allianz-sentiment-challenge/lint.yml?label=flake8&style=flat)](https://github.com/rlacher/allianz-sentiment-challenge/actions/workflows/lint.yml)
[![pytest](https://img.shields.io/github/actions/workflow/status/rlacher/allianz-sentiment-challenge/test.yml?label=pytest&style=flat)](https://github.com/rlacher/allianz-sentiment-challenge/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://spdx.org/licenses/MIT.html)

A FastAPI microservice exposing sentiment analysis on user comments from a mock social platform, Feddit. It fetches comments from a specified subfeddit, classifies their sentiment, and returns structured JSON results.

> **Note:** This project serves as a technical submission for the Allianz Machine Learning Engineer challenge.

## Key Features

- **Fetch Comments:** Retrieves 25 comments from a specified subfeddit.
- **Analyse Sentiment:** Classifies comments as *positive* or *negative* using a sentiment model.
- **Structured Output:** Returns comment text, polarity scores and sentiment labels in JSON format.
- **FastAPI-Based:** Built with a lightweight, high-performance RESTful API framework.
- **Automated Tests:** Include integration tests to verify core API functions.
- **CI Integration:** Runs linting and tests automatically on every commit for code quality assurance.

## Setup

Ensure Python 3.12+ is installed on your system.

### Installation

Follow these steps to set up the project locally.

```bash
# Clone repository and navigate to project root
git clone https://github.com/rlacher/allianz-sentiment-challenge.git
cd allianz-sentiment-challenge

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Install Git pre-commit hooks
pre-commit install
```

### Run the API Server

```bash
uvicorn main:app --reload
```

The server will be accessible at: `http://localhost:8000`.

## API Usage

The service exposes a single GET endpoint at `/api/v1/comments`.

### Example Request

Provide the query parameter `subfeddit_title` to specify the source (e.g. `Dummy Topic 1`).

```bash
curl -X GET "http://localhost:8000/api/v1/comments?subfeddit_title=Dummy%20Topic%201"
```

### Example Response

```json
{
  "title":"Dummy Topic 1",
  "limit":25,
  "comments": [
    {
      "id": "12345",
      "text": "This is amazing!",
      "polarity": 0.9,
      "sentiment": "positive"
    },
    {
      "id": "67890",
      "text": "I hate this!",
      "polarity": -0.8,
      "sentiment": "negative"
    }
  ]
}
```

Additional comments were omitted from the comments array for readability in this example.

## Documentation

Comprehensive Python docstrings are included throughout the codebase for clarity and maintainability.

The project automatically generates an interactive OpenAPI reference via Swagger UI, accessible at ```http://localhost:8000/docs``` when running the service locally.

## Test

Basic automated integration tests verify core API functionality.
Tests cover key functional paths due to the time-constrained scope of this challenge.

Run the following from the project root to execute the test suite with an inlined coverage report:

```bash
pytest --cov=feddit_sentiment --cov-branch tests/
```

The report shows statement and branch coverage.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by [René Lacher](https://github.com/rlacher).
