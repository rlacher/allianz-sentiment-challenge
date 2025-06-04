# Allianz Sentiment Challenge

<!-- Badges -->
[![flake8](https://img.shields.io/github/actions/workflow/status/rlacher/allianz-sentiment-challenge/lint.yaml?label=flake8&style=flat)](https://github.com/rlacher/allianz-sentiment-challenge/actions/workflows/lint.yaml)
[![pytest](https://img.shields.io/github/actions/workflow/status/rlacher/allianz-sentiment-challenge/test.yaml?label=pytest&style=flat)](https://github.com/rlacher/allianz-sentiment-challenge/actions/workflows/test.yaml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://spdx.org/licenses/MIT.html)

A FastAPI microservice exposing sentiment analysis on user comments from a mock social platform, Feddit. It fetches comments from a specified subfeddit, classifies their sentiment, and returns structured JSON results.

> **Note:** This project serves as a technical submission for the Allianz Machine Learning Engineer challenge.

## Key Features

- **Retrieve Comments:** Fetches the 25 most recent comments from a chosen subfeddit.
- **Analyse Sentiment:** Classifies comments as *positive* or *negative* using a sentiment model.
- **Structured Output:** Delivers clear, well-formatted JSON responses with sentiment data.
- **Filtering & Sorting:** Filters comments by time and sorts by polarity or date for tailored results.
- **FastAPI-Based:** Built with a lightweight, high-performance RESTful API framework for scalability.
- **Asynchronous Requests:** Uses async HTTP calls for concurrent fetches, improving response times.
- **Modular Design:** Separates routing and service logic for clarity and testability.
- **Containerised Setup:** Provides a consistent, reproducible setup using Docker Compose.
- **Automated Tests:** Includes unit and integration tests to strengthen API resilience.
- **CI Integration:** Automates linting and testing on every commit, reinforcing code quality.

## Run with Docker (Recommended)

*Prerequisite:* Docker ≥ 23 is installed and running.

Start the sentiment API along with its dependencies (Feddit and PostgreSQL) in a self-contained environment:

```bash
docker compose up --build
```

## Local Setup (Alternative)

*Prerequisite:* Python 3.12+ is installed.

Follow these steps to set up the project locally.

```bash
# Clone repository and navigate to project root
git clone https://github.com/rlacher/allianz-sentiment-challenge.git
cd allianz-sentiment-challenge

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Install dependencies
pip3 install -r requirements.txt

# Install Git pre-commit hooks
pre-commit install
```

Run the API Server locally:

```bash
uvicorn main:app --reload
```

## API Usage

Once running, the API is available at: `http://localhost:8000`.
The service exposes a single GET endpoint at `/api/v1/comments`.

### Query Parameterisation

Provide `subfeddit_title` to specify the source (e.g. `Dummy Topic 1`).

Optionally:
- **Sort by Polarity:** Use `polarity_sort_order` (`asc` or `desc`).  
- **Filter by Time Range:**  
  - `time_from` excludes older comments before a specific timestamp.  
  - `time_to` excludes newer comments beyond a specific timestamp.  
  - Combine both to retrieve comments within a defined range.  

### Example Request

```bash
curl -X GET "http://localhost:8000/api/v1/comments?subfeddit_title=Dummy%20Topic%201"
```

### Example Response

```json
{
  "subfeddit": {
    "id": 1,
    "title": "Dummy Topic 1"
  },
  "comment_count": 25,
  "filter": "None",
  "sort": {
    "key": "created_at",
    "order": "desc"
  },
  "comments": [
    {
      "id": 12345,
      "text": "This is amazing!",
      "created_at": 1748871983,
      "polarity": 0.9,
      "sentiment": "positive"
    },
    {
      "id": 67890,
      "text": "Hate it!",
      "created_at": 1748871984,
      "polarity": -0.8,
      "sentiment": "negative"
    }
  ]
}
```

Additional comments were omitted from the comments array for readability in this example.

## Documentation

Comprehensive Python docstrings are included throughout the codebase for clarity and maintainability.

The project automatically generates an interactive OpenAPI reference via Swagger UI when running the service, accessible at: [http://localhost:8000/docs](http://localhost:8000/docs).

## Test

Unit and integration tests verify the API contract, focusing on critical functionality within the constraints of this challenge.

Run the following from the project root to execute the test suite with an inlined coverage report:

```bash
pytest --cov=feddit_sentiment --cov-branch tests/
```

The report shows statement and branch coverage.

## Technical Notes

The API returns the 25 most recent comments as required.
Internally, all comments are retrieved exhaustively through lazy pagination, optimising resource usage for large datasets.
Comments are sorted by `created_at` in descending order before limiting the output.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by [René Lacher](https://github.com/rlacher).
