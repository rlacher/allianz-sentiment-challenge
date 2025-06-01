# Allianz Sentiment Challenge

This microservice exposes a RESTful API for sentiment analysis of user comments from a mock social media platform, Feddit.
Built using FastAPI, it fetches comments from a specified subfeddit, classifies each and returns a structured JSON response.

> **Note:** This project serves as a technical submission for the Allianz Machine Learning Engineer challenge.

## Key Features

- **Fetch Comments:** Retrieves 25 comments from a given subfeddit.
- **Analyse Sentiment:** Uses a sentiment model to classify each comment as *positive* or *negative*.
- **FastAPI-Based:** Built with a lightweight, high-performant RESTful API framework.
- **Structured Output:** Returns comment text, polarity scores and sentiment classification in JSON format.

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

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by [René Lacher](https://github.com/rlacher).
