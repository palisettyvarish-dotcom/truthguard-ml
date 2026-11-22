import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_api_key = os.getenv("TAVILY_API_KEY")
if not _api_key:
    raise RuntimeError("TAVILY_API_KEY is missing from .env")

client = TavilyClient(api_key=_api_key)


def search_claim(claim: str, max_results: int = 5):
    """Find web evidence relevant to a fact-checking claim."""

    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty")

    response = client.search(
        query=claim,
        search_depth="advanced",
        max_results=max_results,
        topic="general",
    )

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0),
        }
        for item in response.get("results", [])
    ]
