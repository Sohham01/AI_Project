# Ingestion System Documentation

## Overview
The ingestion layer is source-independent, allowing multiple Reddit collection methods to feed the same normalization and persistence pipeline.

## Providers
- **Arctic Shift**: Historical/bulk data and comment retrieval.
- **RSS**: Recent post discovery.
- **Hermes**: Experimental browser/agent-based provider.
- **Decodo**: Optional web-scraping provider.

## Capabilities
- Search Posts
- Get Post
- Get Comments
- Scrape URL
- Discover Targets

## Configuration
Use `.env` for configuration:
- `HERMES_API_URL`
- `HERMES_API_KEY`
- `DECODO_API_TOKEN`

## Running Tests
Run all unit tests using `python -m unittest discover`.
Run integration tests individually.
